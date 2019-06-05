#include "common/common_utils.h"
#include "imgproc/segmentation/permutohedral_common.h"
#include "imgproc/segmentation/image_permutohedral_index.h"
#include <device_launch_parameters.h>
#include <set>

namespace surfelwarp { namespace device {

	__global__ void buildPermutohedralKeysKernel(
		cudaTextureObject_t clip_normalized_rgb_img,
		const float sigma_spatial, const float sigma_rgb,
		const unsigned subsampled_rows, const unsigned subsampled_cols,
		//These three output array should have the same layout
		LatticeCoordKey<image_permutohedral_dim>* lattice_key,
		float* barycentric,
		unsigned* lattice_hashed_key,
		unsigned* lattice_index
	) {
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x >= subsampled_cols || y >= subsampled_rows) return;

		//Query the image and contruct the feature
		const auto rgb_x = crf_subsample_rate * x;
		const auto rgb_y = crf_subsample_rate * y;
		const float4 normalized_rgba = tex2D<float4>(clip_normalized_rgb_img, rgb_x, rgb_y);
		float feature[image_permutohedral_dim];
		feature[0] = float(x) / sigma_spatial;
		feature[1] = float(y) / sigma_spatial;
		feature[2] = normalized_rgba.x * 255.f / sigma_rgb;
		feature[3] = normalized_rgba.y * 255.f / sigma_rgb;
		feature[4] = normalized_rgba.z * 255.f / sigma_rgb;

		//Compute the key and weight
		LatticeCoordKey<image_permutohedral_dim> key_local[image_permutohedral_dim + 1];
		float barycentric_local[image_permutohedral_dim + 2];
		unsigned key_hashed[image_permutohedral_dim + 1];
		permutohedral_lattice(feature, key_local, barycentric_local);

		//Compute the hash value
		for(auto i = 0; i < image_permutohedral_dim + 1; i++) {
			key_hashed[i] = key_local[i].hash();
		}

		//Store the result
		const auto flatten_idx = x + subsampled_cols * y;
		const auto offset = (image_permutohedral_dim + 1) * flatten_idx;
		for(auto i = 0; i < image_permutohedral_dim + 1; i++) {
			lattice_key[offset + i] = key_local[i];
			barycentric[offset + i] = barycentric_local[i];
			lattice_hashed_key[offset + i] = key_hashed[i];
			lattice_index[offset + i] = offset + i;
		}
	}

	__global__ void markUniqueHashedKeyKernel(
		const PtrSz<const unsigned> lattice_hashed_key,
		PtrSz<unsigned> indicator
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < lattice_hashed_key.size) {
			unsigned indicator_value = 0;
			if(idx == 0) {
				indicator_value = 1;
			} else if(lattice_hashed_key[idx] != lattice_hashed_key[idx - 1]) {
				indicator_value = 1;
			}

			//Store to array
			indicator[idx] = indicator_value;
		}
	}


	__global__ void compactUniqueHashedKeyKernel(
		const PtrSz<const unsigned> unique_indicator,
		const unsigned* prefixsum_indicator,
		unsigned* compacted_offset
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < unique_indicator.size) {
			const unsigned flag = unique_indicator[idx];
			if(flag != 0) {
				const unsigned offset = prefixsum_indicator[idx] - 1;
				compacted_offset[offset] = idx;
			}

			//The offset is larger than compacted key of 1
			if(idx == unique_indicator.size - 1) {
				compacted_offset[prefixsum_indicator[idx]] = unique_indicator.size;
			}
		}
	}

}; /* End of namespace device */
}; /* End of namespace surfelwarp */

void surfelwarp::buildPermutohedralKeys(
	cudaTextureObject_t clip_normalized_rgb_img, 
	const float sigma_spatial, const float sigma_rgb,
	const unsigned subsampled_rows, const unsigned subsampled_cols,
	DeviceArray<LatticeCoordKey<image_permutohedral_dim>>& lattice_key,
	DeviceArray<float>& barycentric, 
	DeviceArray<unsigned>& lattice_hashed_key,
	DeviceArray<unsigned>& lattice_index,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(subsampled_cols, blk.x), divUp(subsampled_rows, blk.y));
	device::buildPermutohedralKeysKernel<<<grid, blk, 0, stream>>>(
		clip_normalized_rgb_img, 
		sigma_spatial, sigma_rgb, 
		subsampled_rows, subsampled_cols, 
		lattice_key, 
		barycentric, 
		lattice_hashed_key, 
		lattice_index
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::markUniqueHashedKey(
	const DeviceArray<unsigned>& sorted_lattice_hashed_key,
	DeviceArray<unsigned>& indicator, 
	cudaStream_t stream
) {
	dim3 blk(128);
	dim3 grid(divUp(sorted_lattice_hashed_key.size(), blk.x));
	device::markUniqueHashedKeyKernel<<<grid, blk, 0, stream>>>(
		sorted_lattice_hashed_key,
		indicator
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}



void surfelwarp::compactUniqueHashKey(
	const DeviceArray<unsigned>& unique_indicator, 
	const DeviceArray<unsigned>& prefixsum_indicator, 
	DeviceArray<unsigned>& compacted_offset,
	cudaStream_t stream
) {
	dim3 blk(128);
	dim3 grid(divUp(unique_indicator.size(), blk.x));
	device::compactUniqueHashedKeyKernel<<<grid, blk, 0, stream>>>(
		unique_indicator, 
		prefixsum_indicator,
		compacted_offset
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}