#include "common/Constants.h"
#include "common/ConfigParser.h"
#include "common/Stream.h"
#include "common/Serializer.h"
#include "common/BinaryFileStream.h"
#include "common/sanity_check.h"
#include "imgproc/correspondence/PatchColliderForest.h"
#include "imgproc/correspondence/PatchColliderRGBCorrespondence.h"
#include "imgproc/correspondence/gpc_feature.h"

#include <device_launch_parameters.h>
#include <boost/mpl/min_max.hpp>


namespace surfelwarp { namespace device {

	template<int FeatureDim, int NumTrees>
	__device__ __forceinline__ unsigned searchGPCForest(
		const GPCPatchFeature<FeatureDim>& feature,
		const typename PatchColliderForest<FeatureDim, NumTrees>::GPCForestDevice& forest
	) {
		unsigned hash = 0;
		for(auto i = 0; i < NumTrees; i++) {
			const GPCTree<FeatureDim>& tree = forest.trees[i];
			const unsigned leaf = tree.leafForPatch(feature);
			hash = hash * 67421 + leaf;
		}
		return hash;
	}

	__host__ __device__ __forceinline__
	unsigned encode_pixel_impair(int rgb_x, int rgb_y, bool img_0) {
		unsigned encoded = rgb_x + 1024 * rgb_y;
		if(img_0) {
			encoded = encoded & (~(1 << 31));
		} 
		else {
			encoded = encoded | (1 << 31);
		}
		return encoded;
	}

	__host__ __device__ __forceinline__
	void decode_pixel_impair(
		unsigned encoded, 
		int& rgb_x, int& rgb_y, 
		bool& img_0
	) {
		//Check where this pixel is from
		if ((encoded & (1 << 31)) != 0) {
			img_0 = false;
		} 
		else {
			img_0 = true;
		}

		//Zero out highest bit
		encoded = encoded & (~(1 << 31));
		rgb_x = encoded % 1024;
		rgb_y = encoded / 1024;
	}
	
	template<int PatchHalfSize, int NumTrees>
	__global__ void buildColliderKeyValueKernel(
		cudaTextureObject_t rgb_0, cudaTextureObject_t rgb_1,
		const typename PatchColliderForest<18, NumTrees>::GPCForestDevice forest,
		const int stride, const int kv_rows, const int kv_cols,
		unsigned* keys, unsigned* values
	) {
		const auto kv_x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto kv_y = threadIdx.y + blockDim.y * blockIdx.y;
		if(kv_x >= kv_cols || kv_y >= kv_rows) return;

		//Transfer to the center of rgb image
		const auto rgb_center_x = PatchHalfSize + kv_x * stride;
		const auto rgb_center_y = PatchHalfSize + kv_y * stride;

		//Build the feature
		GPCPatchFeature<18> patch_feature_0, patch_feature_1;
		buildDCTPatchFeature<PatchHalfSize>(rgb_0, rgb_center_x, rgb_center_y, patch_feature_0);
		buildDCTPatchFeature<PatchHalfSize>(rgb_1, rgb_center_x, rgb_center_y, patch_feature_1);

		//Search it for the key
		const unsigned key_0 = searchGPCForest<18, NumTrees>(patch_feature_0, forest);
		const unsigned key_1 = searchGPCForest<18, NumTrees>(patch_feature_1, forest);

		//Build the value
		const unsigned value_0 = encode_pixel_impair(rgb_center_x, rgb_center_y, true);
		const unsigned value_1 = encode_pixel_impair(rgb_center_x, rgb_center_y, false);

		//Store it
		const auto offset = 2 * (kv_x + kv_cols * kv_y);
		keys[offset + 0] = key_0;
		keys[offset + 1] = key_1;
		values[offset + 0] = value_0;
		values[offset + 1] = value_1;
	}


	__global__ void markCorrespondenceCandidateKernel(
		const PtrSz<const unsigned> sorted_treeleaf_key,
		const unsigned* sorted_pixel_value,
		cudaTextureObject_t foreground_1,
		unsigned* candidate_indicator
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx >= sorted_treeleaf_key.size) return;

		//The indicator must be written
		unsigned is_candidate = 0;

		//Check idx is the first index of some key
		if(idx == 0 || sorted_treeleaf_key[idx] != sorted_treeleaf_key[idx - 1]) {
			//Read the value
			const auto hashed_key = sorted_treeleaf_key[idx];
			
			//Count the number of matching
			int num_pixels_keys = 1;

			//The end of probing
			auto end = idx + 2;
			if(end >= sorted_treeleaf_key.size) 
				end = sorted_treeleaf_key.size - 1;
			
			//Probe the next
			for(int j = idx + 1; j <= end; j++) {
				if(sorted_treeleaf_key[j] == hashed_key) 
					num_pixels_keys++;
			}

			//Determine whether the pixel are from different img
			if(num_pixels_keys == 2) {
				int x, y;
				bool pixel0_img0, pixel1_img0;
				const auto encoded_pixel_0 = sorted_pixel_value[idx + 0];
				//Now we are safe to read the idx + 1 without checking
				const auto encoded_pixel_1 = sorted_pixel_value[idx + 1];
				decode_pixel_impair(encoded_pixel_0, x, y, pixel0_img0);
				decode_pixel_impair(encoded_pixel_1, x, y, pixel1_img0);

				//If the from different image
				if((pixel0_img0 && (!pixel1_img0)) || ((!pixel0_img0) && pixel1_img0)  ) {
					//Determine which one is for image 1
					if(!pixel0_img0) {
						decode_pixel_impair(encoded_pixel_0, x, y, pixel0_img0);
					} 
					else {
						decode_pixel_impair(encoded_pixel_1, x, y, pixel1_img0);
					}
					
					//Check if this is foreground
					const unsigned char is_foreground = tex2D<unsigned char>(foreground_1, x, y);
					if(is_foreground != 0)
						is_candidate = 1;
				}
			}
		} // Ensure this idx is the first one of some new key

		//Write it
		candidate_indicator[idx] = is_candidate;
	}


	__global__ void collectCandidatePixelPairKernel(
		const PtrSz<const unsigned> candidate_indicator,
		const unsigned* sorted_pixel_value,
		const unsigned* prefixsum_indicator,
		//The output
		ushort4* pixel_pair_array
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx >= candidate_indicator.size) return;

		//For any valid indicator, it is safe to read its sucessor
		if(candidate_indicator[idx] > 0) {
			ushort4 pixel_pair;
			int x, y;
			bool img_0;

			//These read must be safe
			const auto encoded_pixel_0 = sorted_pixel_value[idx + 0];
			const auto encoded_pixel_1 = sorted_pixel_value[idx + 1];

			//Decode and write
			decode_pixel_impair(encoded_pixel_0, x, y, img_0);
			if(img_0) {
				pixel_pair.x = x;
				pixel_pair.y = y;
			} 
			else {
				pixel_pair.z = x;
				pixel_pair.w = y;
			}

			decode_pixel_impair(encoded_pixel_1, x, y, img_0);
			if(img_0) {
				pixel_pair.x = x;
				pixel_pair.y = y;
			} 
			else {
				pixel_pair.z = x;
				pixel_pair.w = y;
			}
			
			//Write it
			const auto offset = prefixsum_indicator[idx] - 1;
			pixel_pair_array[offset] = pixel_pair;
		}
	}

}; // namespace device
}; // namespace surfelwarp

surfelwarp::PatchColliderRGBCorrespondence::PatchColliderRGBCorrespondence()
{
	m_rgb_cols = m_rgb_rows = 0;
}

void surfelwarp::PatchColliderRGBCorrespondence::AllocateBuffer(unsigned img_rows, unsigned img_cols)
{
	//Read the gpc model and upload to device
	//BinaryFileStream in_fstream(Constants::kGPCModelPath.c_str(), BinaryFileStream::Mode::ReadOnly);
	const auto& config = ConfigParser::Instance();
	BinaryFileStream in_fstream(config.gpc_model_path().string().c_str(), BinaryFileStream::Mode::ReadOnly);
	m_forest.Load(&in_fstream);
	m_forest.UploadToDevice();

	//Restrict the maximum level?
	m_forest.UpdateSearchLevel(max_search_level);

	//Determine the size of key-value map
	m_rgb_rows = img_rows;
	m_rgb_cols = img_cols;
	m_kvmap_rows = (m_rgb_rows - patch_clip * 2) / patch_stride; // + 1;
	m_kvmap_cols = (m_rgb_cols - patch_clip * 2) / patch_stride; // + 1;

	//Allocate the key-value buffer
	const auto kv_size = m_kvmap_rows * m_kvmap_rows;
	
	//Both rgb_0 and rgb_1 will have key-value pairs
	m_treeleaf_key.create(2 * kv_size);
	m_pixel_value.create(2 * kv_size);
	m_collide_sort.AllocateBuffer(2 * kv_size);

	//The buffer for marking the valid indicator
	m_candidate_pixelpair_indicator.create(2 * kv_size);
	
	//The buffer for prefixsum and compaction
	m_prefixsum.AllocateBuffer(m_candidate_pixelpair_indicator.size());
	cudaSafeCall(cudaMallocHost((void**)(&m_candidate_size_pagelock), sizeof(unsigned)));

	//The buffer for output
	m_correspondence_pixels.AllocateBuffer(max_num_correspondence);
}

void surfelwarp::PatchColliderRGBCorrespondence::ReleaseBuffer()
{
	//Clear the key-value pair
	m_treeleaf_key.release();
	m_pixel_value.release();
	
	//Clear the indicator
	m_candidate_pixelpair_indicator.release();
	
	//Clear the pagelock memory
	cudaSafeCall(cudaFreeHost(m_candidate_size_pagelock));
}

void surfelwarp::PatchColliderRGBCorrespondence::SetInputImages(
	cudaTextureObject_t rgb_0,
	cudaTextureObject_t rgb_1, 
	cudaTextureObject_t foreground_1
) {
	rgb_0_ = rgb_0;
	rgb_1_ = rgb_1;
	m_foreground_1 = foreground_1;
}

void surfelwarp::PatchColliderRGBCorrespondence::SetInputImages(
	cudaTextureObject_t rgb_0, cudaTextureObject_t rgb_1, 
	cudaTextureObject_t depth_0, cudaTextureObject_t depth_1
) {
	throw new std::runtime_error("This version of patch collider only accept rgb images");
}

void surfelwarp::PatchColliderRGBCorrespondence::FindCorrespondence(cudaStream_t stream)
{
	dim3 kv_blk(8, 8);
	dim3 kv_grid(divUp(m_kvmap_cols, kv_blk.x), divUp(m_kvmap_rows, kv_blk.y));
	const auto forest_dev = m_forest.OnDevice();
	device::buildColliderKeyValueKernel<patch_radius, num_trees><<<kv_grid, kv_blk, 0, stream>>>(
		rgb_0_, rgb_1_, 
		forest_dev, 
		patch_stride, 
		m_kvmap_rows, m_kvmap_cols,
		m_treeleaf_key.ptr(),
		m_pixel_value.ptr()
	);
	
	//Sort it
	m_collide_sort.Sort(m_treeleaf_key, m_pixel_value, stream);

	//Debug code
	//std::cout << "The number of unique elments " << numUniqueElement(m_treeleaf_key, 0xffffffffu) << std::endl;
	
	//Mark of candidate
	dim3 indicator_blk(64);
	dim3 indicator_grid(divUp(m_collide_sort.valid_sorted_key.size(), indicator_blk.x));
	device::markCorrespondenceCandidateKernel<<<indicator_grid, indicator_blk, 0, stream>>>(
		m_collide_sort.valid_sorted_key, 
		m_collide_sort.valid_sorted_value, 
		m_foreground_1,
		m_candidate_pixelpair_indicator.ptr()
	);
	
	//Do a prefix-sum
	m_prefixsum.InclusiveSum(m_candidate_pixelpair_indicator, stream);
	
	//Get the size of sum
	cudaSafeCall(cudaMemcpyAsync(
		m_candidate_size_pagelock,
		m_prefixsum.valid_prefixsum_array + m_prefixsum.valid_prefixsum_array.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream
	));
	
	//Invoke the collection kernel
	device::collectCandidatePixelPairKernel<<<indicator_grid, indicator_blk, 0, stream>>>(
		m_candidate_pixelpair_indicator, 
		m_collide_sort.valid_sorted_value.ptr(), 
		m_prefixsum.valid_prefixsum_array.ptr(),
		m_correspondence_pixels.Ptr()
	);
	
	//Construct the output
	cudaSafeCall(cudaStreamSynchronize(stream));
	m_correspondence_pixels.ResizeArrayOrException(*m_candidate_size_pagelock);

	//Debug
	//std::cout << "The number of candidate array is " << m_valid_correspondence_array.size() << std::endl;

	//Check here
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

