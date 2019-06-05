#include "common/Constants.h"
#include "common/common_texture_utils.h"
#include "imgproc/segmentation/ForegroundSegmenterPermutohedral.h"
#include "imgproc/segmentation/image_permutohedral_index.h"
#include "imgproc/segmentation/crf_common.h"
#include "imgproc/segmentation/permutohedral_common.h"
#include "visualization/Visualizer.h"

#include <algorithm>
#include <iostream>
#include <device_launch_parameters.h>
#include <common/Constants.h>
#include "hashing/TicketBoardSet.cuh"


namespace surfelwarp { namespace device {
	

	__global__ void buildLatticeIndexKernel(
		cudaTextureObject_t rgb_image,
		const unsigned subsampled_rows, const unsigned subsampled_cols,
		//Normalizing constants
		const float sigma_alpha, const float sigma_beta,
		//The information for hash table
		unsigned* ticket_board, 
		LatticeCoordKey<5>* table, const unsigned table_size,
		const uint2 primary_hash, const uint2 step_hash
	) {
		//Compute the position
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x >= subsampled_cols || y >= subsampled_rows) return;

		//Construct the feature vector
		const auto rgb_x = crf_subsample_rate * x;
		const auto rgb_y = crf_subsample_rate * y;
		const float4 normalized_rgba = tex2D<float4>(rgb_image, rgb_x, rgb_y);

		//Construct the feature for this pixel
		float feature[5];
		feature[0] = float(x) / sigma_alpha;
		feature[1] = float(y) / sigma_alpha;
		feature[2] = normalized_rgba.x * 255.f / sigma_beta;
		feature[3] = normalized_rgba.y * 255.f / sigma_beta;
		feature[4] = normalized_rgba.z * 255.f / sigma_beta;

		//Compute the lattice key
		LatticeCoordKey<5> lattice_coord_keys[6];
		float lattice_weights[7];
		permutohedral_lattice(feature, lattice_coord_keys, lattice_weights);

		//Insert into the hash table
		for(auto i = 0; i < 6; i++) {
			const auto hashed_lattice = lattice_coord_keys[i].hash();
			hashing::device::insertTicketSetEntry(
				lattice_coord_keys[i], hashed_lattice, 
				ticket_board, table, table_size, 
				primary_hash, step_hash
			);
		}
	}


	enum {
		kSplatBlockDim = 8,
		kSplatBlockSize = kSplatBlockDim * kSplatBlockDim,
	};

	__global__ void foregroundPermutohedralSplatKernel(
		//The input maps
		cudaTextureObject_t meanfield_foreground_in,
		cudaTextureObject_t rgb_image,
		const unsigned subsampled_rows, const unsigned subsampled_cols,
		//Normalizing constants
		const float sigma_alpha, const float sigma_beta,
		//The hash table attributes
		const typename hashing::TicketBoardSet<LatticeCoordKey<5>>::Device lattice_set,
		//The splat value
		PtrSz<float2> lattice_value_array
	) {
		//Compute the position
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x >= subsampled_cols || y >= subsampled_rows) return;

		//Construct the feature vector
		const auto rgb_x = crf_subsample_rate * x;
		const auto rgb_y = crf_subsample_rate * y;
		const float4 normalized_rgba = tex2D<float4>(rgb_image, rgb_x, rgb_y);

		//Construct the feature for this pixel
		float feature[5];
		feature[0] = float(x) / sigma_alpha;
		feature[1] = float(y) / sigma_alpha;
		feature[2] = normalized_rgba.x * 255.f / sigma_beta;
		feature[3] = normalized_rgba.y * 255.f / sigma_beta;
		feature[4] = normalized_rgba.z * 255.f / sigma_beta;

		//Compute the lattice key
		LatticeCoordKey<5> lattice_coord_keys[6];
		float lattice_weights[7];
		permutohedral_lattice(feature, lattice_coord_keys, lattice_weights);

		//The shared index array
		__shared__ unsigned lattice_index[6 * kSplatBlockSize];

		//Query the index in the compacted array
		const unsigned inblock_offset = threadIdx.x + kSplatBlockDim * threadIdx.y;
		unsigned* compacted_index = lattice_index + 6 * inblock_offset;
		for(auto i = 0; i < 6; i++) {
			const auto hashed_lattice = lattice_coord_keys[i].hash();
			compacted_index[i] = hashing::device::retrieveTicketSetKeyIndex<LatticeCoordKey<5>>(
				lattice_coord_keys[i], hashed_lattice, 
				lattice_set.ticket_board, lattice_set.table, 
				lattice_set.table_size, 
				lattice_set.primary_hash, lattice_set.step_hash
			);

			//Debug
			//if(compacted_index[i] == 0xffffffffu) {
			//	printf("Incorrect retrieve\n");
			//}
		}

		//Compute the energy for this pixel
		const float prev_foreground_prob = tex2D<float>(meanfield_foreground_in, x, y);
		const float prev_backround_prob = 1.0f - prev_foreground_prob;
		__shared__ float2 lattice_energy[6 * kSplatBlockSize];
		float2* energy_thread = lattice_energy + 6 * inblock_offset;
		for(auto i = 0; i < 6; i++) {
			energy_thread[i].x = prev_backround_prob * lattice_weights[i];
			energy_thread[i].y = prev_foreground_prob * lattice_weights[i];
		}

		//Sync threads here
		__syncthreads();

		for(auto i = 0; i < 6; i++) {
			const auto curr_lattice = compacted_index[i];
			float2 energy = make_float2(0.0f, 0.0f);
			bool write_to = true;

			//The loop to iterate through the shared memory
			//The energy of this thread is also counted here
			for(auto j = 0; j < 6 * kSplatBlockSize; j++) {
				if(curr_lattice == lattice_index[j]) {
					energy.x += lattice_energy[j].x;
					energy.y += lattice_energy[j].y;
					const auto j_thread = j / 6;
					if(j_thread < inblock_offset) write_to = false;
				}
			}

			//write to the global memory if required
			if(write_to && curr_lattice < lattice_value_array.size) {
				float* foreground_energy_pos = &(lattice_value_array[curr_lattice].x);
				float* background_energy_pos = foreground_energy_pos + 1;
				atomicAdd(foreground_energy_pos, energy.x);
				atomicAdd(background_energy_pos, energy.y);
			}
		} // energy adding loop
	} // energy splat kernel


	__global__ void foregroundPermutohedralSliceKernel(
		cudaTextureObject_t meanfield_foreground_in,
		cudaTextureObject_t rgb_image,
		PtrStepSz<const float2> unary_energy_map,
		//Normalizing constants
		const float sigma_alpha, 
		const float sigma_beta,
		const float sigma_gamma,
		//The weight constants
		const float appearance_weight,
		const float smooth_weight,
		//The search structure and compacted value
		const typename hashing::TicketBoardSet<LatticeCoordKey<5>>::Device lattice_set,
		const PtrSz<const float2> lattice_energy,
		//The output
		cudaSurfaceObject_t meanfield_foreground_out
	) {
		//Compute the position
		const int x = threadIdx.x + blockDim.x * blockIdx.x;
		const int y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x >= unary_energy_map.cols || y >= unary_energy_map.rows) return;

		//Construct the feature vector
		const int rgb_x = crf_subsample_rate * x;
		const int rgb_y = crf_subsample_rate * y;
		const float4 normalized_rgba = tex2D<float4>(rgb_image, rgb_x, rgb_y);

		//Construct the feature for this pixel
		float feature[5];
		feature[0] = float(x) / sigma_alpha;
		feature[1] = float(y) / sigma_alpha;
		feature[2] = normalized_rgba.x * 255.f / sigma_beta;
		feature[3] = normalized_rgba.y * 255.f / sigma_beta;
		feature[4] = normalized_rgba.z * 255.f / sigma_beta;

		//Compute the lattice key
		LatticeCoordKey<5> lattice_coord_keys[6];
		float lattice_weights[7];
		permutohedral_lattice(feature, lattice_coord_keys, lattice_weights);

		//Collect the energy from lattice
		float e_foreground = 0.0f, e_background = 0.0f;
		for(auto i = 0; i < 6; i++) {
			const auto hashed_lattice = lattice_coord_keys[i].hash();

			//the index in the compacted array
			const auto compacted_index = hashing::device::retrieveTicketSetKeyIndex<LatticeCoordKey<5>>(
				lattice_coord_keys[i], hashed_lattice, 
				lattice_set.ticket_board, lattice_set.table, 
				lattice_set.table_size, 
				lattice_set.primary_hash, lattice_set.step_hash
			);

			//Collect the energy
			if(compacted_index < lattice_energy.size) {
				const float2 energy = lattice_energy[compacted_index];
				const float weight = appearance_weight;
				e_foreground += weight * lattice_weights[i] * energy.x;
				e_background += weight * lattice_weights[i] * energy.y;
			}
		}

		//Collect the smooth energy
		const int halfsize = 7;
		for(int neighbor_y = y - halfsize; neighbor_y <= y + halfsize; neighbor_y++) {
			for(int neighbor_x = x - halfsize; neighbor_x <= x + halfsize; neighbor_x++) {
				//Compute the kernel value
				const float kernel_value = smooth_weight * smooth_kernel(x, y, neighbor_x, neighbor_y, sigma_gamma);
				
				//Message passing
				const float neighbor_foreground_prob = tex2D<float>(meanfield_foreground_in, neighbor_x, neighbor_y);
				const float neighbor_backround_prob = 1.0f - neighbor_foreground_prob;

				//Note that the window might be outside, in that case, tex2D should return zero
				if(neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < unary_energy_map.cols && neighbor_y < unary_energy_map.rows) {
					e_foreground += (neighbor_backround_prob * kernel_value);
					e_background += (neighbor_foreground_prob * kernel_value);
				}
			}
		}// Collect of the smooth kernel

		// subtract self-energy
		const float prev_foreground_prob = tex2D<float>(meanfield_foreground_in, x, y);
		const float prev_backround_prob = 1.0f - prev_foreground_prob;
		e_foreground -= prev_backround_prob * (appearance_weight + smooth_weight);
		e_background -= prev_foreground_prob * (appearance_weight + smooth_weight);

		//Update the mean field locally
		const float2 unary_energy = unary_energy_map.ptr(y)[x];
		const float foreground_energy = unary_energy.x + e_foreground;
		const float background_energy = unary_energy.y + e_background;
		const float energy_diff = foreground_energy - background_energy;

		//Note the numerical problem involved with expf
		float foreground_prob;
		const float energy_cutoff = 20.0f;
		if(energy_diff < - energy_cutoff) {
			foreground_prob = 1.0f;
		} else if(energy_diff > energy_cutoff) {
			foreground_prob = 0.0f;
		} else {
			const float exp_energy_diff = __expf(energy_diff);
			foreground_prob = 1.0f / (1.0f + exp_energy_diff);
		}

		//Well there might be numerical errors
		if (foreground_prob > 1.0f) {
			foreground_prob = 1.0f;
		} else if(foreground_prob < 0.0f) {
			foreground_prob = 1e-3f;
		}

		//Write to the surface
		surf2Dwrite(foreground_prob, meanfield_foreground_out, x * sizeof(float), y);
	} //The slice kernel


}; // namespace device
}; // namespace surfelwarp


void surfelwarp::ForegroundSegmenterPermutohedral::AllocateBuffer(
	unsigned clip_rows, unsigned clip_cols
) {
	//Do subsampling here
	const auto subsampled_rows = clip_rows / crf_subsample_rate;
	const auto subsampled_cols = clip_cols / crf_subsample_rate;
	
	//Allocate the buffer for meanfield q
	createFloat1TextureSurface(subsampled_rows, subsampled_cols, m_meanfield_foreground_collect_subsampled[0]);
	createFloat1TextureSurface(subsampled_rows, subsampled_cols, m_meanfield_foreground_collect_subsampled[1]);
	
	//Allocate the buffer for unary energy
	m_unary_energy_map_subsampled.create(subsampled_rows, subsampled_cols);
	
	//Allocate the buffer for segmentation mask
	createUChar1TextureSurface(subsampled_rows, subsampled_cols, m_segmented_mask_collect_subsampled);
	
	//Allocate sub-buffers
	allocateLatticeIndexBuffer();
	allocateLatticeValueBuffer();
	
	//Allocate the upsampled buffer
	createUChar1TextureSurface(clip_rows, clip_cols, m_foreground_mask_collect_upsampled);
	createFloat1TextureSurface(clip_rows, clip_cols, m_filter_foreground_mask_collect_upsampled);
}


void surfelwarp::ForegroundSegmenterPermutohedral::ReleaseBuffer()
{
	releaseTextureCollect(m_meanfield_foreground_collect_subsampled[0]);
	releaseTextureCollect(m_meanfield_foreground_collect_subsampled[1]);
	releaseTextureCollect(m_segmented_mask_collect_subsampled);
	m_unary_energy_map_subsampled.release();
	
	//Release other buffers
	releaseLatticeIndexBuffer();
	releaseLatticeValueBuffer();
}


void surfelwarp::ForegroundSegmenterPermutohedral::SetInputImages(
	cudaTextureObject_t clip_normalized_rgb_img, 
	cudaTextureObject_t raw_depth_img, 
	cudaTextureObject_t clip_depth_img,
	int frame_idx,
	cudaTextureObject_t clip_background_rgb
) {
	m_input_texture.clip_normalize_rgb_img = clip_normalized_rgb_img;
	m_input_texture.raw_depth_img = raw_depth_img;
	m_input_texture.clip_depth_img = clip_depth_img;
}


void surfelwarp::ForegroundSegmenterPermutohedral::Segment(cudaStream_t stream)
{
	//Init the mean field
	initMeanfieldUnaryEnergy(stream);
	
	//Build the index
	buildLatticeIndex(stream);
	
	//The inference loop
	const auto max_iters = Constants::kMeanfieldSegmentIteration;
	for(auto i = 0; i < max_iters; i++) {
		//Debug
		//saveMeanfieldApproximationMap(i);
		
		//The inference iters
		splatEnergy(stream);
		slice(stream);
	}
	
	//Write to the segmentation mask
	writeSegmentationMask(stream);
	upsampleFilterForegroundMask(stream);
}

cudaTextureObject_t surfelwarp::ForegroundSegmenterPermutohedral::ForegroundMask() const {
	return m_foreground_mask_collect_upsampled.texture;
}

cudaTextureObject_t surfelwarp::ForegroundSegmenterPermutohedral::FilterForegroundMask() const {
	return m_filter_foreground_mask_collect_upsampled.texture;
}

cudaTextureObject_t surfelwarp::ForegroundSegmenterPermutohedral::SubsampledForegroundMask() const {
	return m_segmented_mask_collect_subsampled.texture;
}

void surfelwarp::ForegroundSegmenterPermutohedral::initMeanfieldUnaryEnergy(cudaStream_t stream) {
	initMeanfieldUnaryForegroundSegmentation(
		m_input_texture.raw_depth_img,
		m_input_texture.clip_depth_img,
		m_unary_energy_map_subsampled,
		m_meanfield_foreground_collect_subsampled[0].surface,
		stream
	);
	m_updated_meanfield_idx = 0;
}


/* Method to build the hash index of lattice coordinate
 */
void surfelwarp::ForegroundSegmenterPermutohedral::allocateLatticeIndexBuffer() {
	//The size of this set is almost emperical
	m_lattice_set.AllocateBuffer(kMaxUniqueLattices);
}



void surfelwarp::ForegroundSegmenterPermutohedral::releaseLatticeIndexBuffer() {
	m_lattice_set.ReleaseBuffer();
}


void surfelwarp::ForegroundSegmenterPermutohedral::buildLatticeIndex(
	cudaStream_t stream
) {
	//Reset the table
	LatticeCoordKey<5> empty; empty.set_null();
	m_lattice_set.ResetTable(empty, stream);

	//Construct the size
	const unsigned subsampled_rows = m_unary_energy_map_subsampled.rows();
	const unsigned subsampled_cols = m_unary_energy_map_subsampled.cols();
	dim3 blk(8, 8);
	dim3 grid(divUp(subsampled_cols, blk.x), divUp(subsampled_rows, blk.y));

	//Invoke the insert kernel
	device::buildLatticeIndexKernel<<<grid, blk, 0, stream>>>(
		//The image information
		m_input_texture.clip_normalize_rgb_img, 
		subsampled_rows, subsampled_cols, 
		//The gaussian sigma
		sigma_alpha_, sigma_beta_, 
		//The hash table information
		m_lattice_set.TicketBoard(), 
		m_lattice_set.Table(), 
		m_lattice_set.TableSize(),
		m_lattice_set.PrimaryHash(), 
		m_lattice_set.StepHash()
	);

	//Build index on the lattice set
	m_lattice_set.BuildIndex(stream);

	//Debug
	//m_lattice_set.IndexInformation();
}


/* The method to perform splat
 */
void surfelwarp::ForegroundSegmenterPermutohedral::allocateLatticeValueBuffer() {
	m_lattice_energy_array.create(kMaxUniqueLattices);
}


void surfelwarp::ForegroundSegmenterPermutohedral::releaseLatticeValueBuffer() {
	m_lattice_energy_array.release();
}

void surfelwarp::ForegroundSegmenterPermutohedral::splatEnergy(cudaStream_t stream)
{
	//First clear the value
	cudaSafeCall(cudaMemsetAsync(
		m_lattice_energy_array.ptr(), 0,
		m_lattice_energy_array.size() * sizeof(float2),
		stream
	));

	//Construct the size
	const unsigned subsampled_rows = m_unary_energy_map_subsampled.rows();
	const unsigned subsampled_cols = m_unary_energy_map_subsampled.cols();
	dim3 blk(device::kSplatBlockDim, device::kSplatBlockDim);
	dim3 grid(divUp(subsampled_cols, blk.x), divUp(subsampled_rows, blk.y));

	//Constrcuct the device hash
	const auto device_set = m_lattice_set.OnDevice();

	//Invoke the kernel
	device::foregroundPermutohedralSplatKernel<<<grid, blk, 0, stream>>>(
		m_meanfield_foreground_collect_subsampled[m_updated_meanfield_idx].texture,
		m_input_texture.clip_normalize_rgb_img, 
		subsampled_rows, subsampled_cols, 
		sigma_alpha_, sigma_beta_, 
		device_set, 
		m_lattice_energy_array
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::ForegroundSegmenterPermutohedral::slice(cudaStream_t stream)
{
	//Constrcuct the device hash
	const auto device_set = m_lattice_set.OnDevice();

	//The output index
	const auto meanfield_input_idx = m_updated_meanfield_idx;
	const auto meanfield_output_idx = (m_updated_meanfield_idx + 1) % 2;

	//Local constants
	const float sigma_gamma = 3;
	const float apperance_weight = 1.0f;
	const float smooth_weight = 0.5f;

	//Invoke the kernel
	const unsigned subsampled_rows = m_unary_energy_map_subsampled.rows();
	const unsigned subsampled_cols = m_unary_energy_map_subsampled.cols();
	dim3 blk(8, 8);
	dim3 grid(divUp(subsampled_cols, blk.x), divUp(subsampled_rows, blk.y));
	device::foregroundPermutohedralSliceKernel<<<grid, blk, 0, stream>>>(
		m_meanfield_foreground_collect_subsampled[meanfield_input_idx].texture,
		m_input_texture.clip_normalize_rgb_img,
		m_unary_energy_map_subsampled,
		sigma_alpha_, sigma_beta_, sigma_gamma, 
		apperance_weight, smooth_weight, 
		device_set, 
		m_lattice_energy_array,
		m_meanfield_foreground_collect_subsampled[meanfield_output_idx].surface
	);

	//Update the index here?
	m_updated_meanfield_idx = meanfield_output_idx;

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::ForegroundSegmenterPermutohedral::writeSegmentationMask(
	cudaStream_t stream
) {
	const auto write_idx = m_updated_meanfield_idx % 2;
	writeForegroundSegmentationMask(
		m_meanfield_foreground_collect_subsampled[write_idx].texture,
		m_unary_energy_map_subsampled.rows(), m_unary_energy_map_subsampled.cols(),
		m_segmented_mask_collect_subsampled.surface,
		stream
	);
}

void surfelwarp::ForegroundSegmenterPermutohedral::upsampleFilterForegroundMask(cudaStream_t stream) {
	ForegroundSegmenter::UpsampleFilterForegroundMask(
		m_segmented_mask_collect_subsampled.texture,
		m_unary_energy_map_subsampled.rows(), m_unary_energy_map_subsampled.cols(),
		crf_subsample_rate,
		Constants::kForegroundSigma,
		m_foreground_mask_collect_upsampled.surface,
		m_filter_foreground_mask_collect_upsampled.surface,
		stream
	);
}

void surfelwarp::ForegroundSegmenterPermutohedral::saveMeanfieldApproximationMap(const unsigned int iter) {
	std::stringstream ss;
	ss << iter;
	std::string file_name = "meanfield-";
	file_name += ss.str();
	file_name += ".png";
	Visualizer::SaveBinaryMeanfield(m_meanfield_foreground_collect_subsampled[m_updated_meanfield_idx].texture, file_name);
}




