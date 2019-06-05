#include "imgproc/segmentation/ForegroundSegmenter.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__global__ void upsampleFilterForegroundMaskKernel(
		cudaTextureObject_t subsampled_mask, 
		unsigned upsample_rows, unsigned upsample_cols,
		unsigned sample_rate,
		const float sigma,
		cudaSurfaceObject_t upsampled_mask, 
		cudaSurfaceObject_t filter_mask
	) {
		const int x = threadIdx.x + blockDim.x * blockIdx.x;
		const int y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x >= upsample_cols || y >= upsample_rows) return;

		//A window search
		const int halfsize = __float2uint_ru(sigma) * 2;
		float total_weight = 0.0f;
		float total_value = 0.0f;
		for(int neighbor_y = y - halfsize; neighbor_y <= y + halfsize; neighbor_y++) {
			for(int neighbor_x = x - halfsize; neighbor_x <= x + halfsize; neighbor_x++) {
				//Retrieve the mask value at neigbour
				const auto subsampled_neighbor_x = neighbor_x / sample_rate;
				const auto subsampled_neighbor_y = neighbor_y / sample_rate;
				const unsigned char neighbor_foreground = tex2D<unsigned char>(subsampled_mask, subsampled_neighbor_x, subsampled_neighbor_y);

				//Compute the gaussian weight
				const float diff_x_square = (neighbor_x - x) * (neighbor_x - x);
				const float diff_y_square = (neighbor_y - y) * (neighbor_y - y);
				const float weight = __expf(0.5f * (diff_x_square + diff_y_square) / (sigma * sigma));

				//Accumlate it
				if(neighbor_x >= 0 && neighbor_x < upsample_cols && neighbor_y >= 0 && neighbor_y < upsample_rows)
				{
					total_weight += weight;
					total_value += weight * float(1 - neighbor_foreground);
				}
			}
		}

		
		//Compute the value locally
		const auto subsampled_x = x / sample_rate;
		const auto subsampled_y = y / sample_rate;
		const unsigned char foreground_indicator = tex2D<unsigned char>(subsampled_mask, subsampled_x, subsampled_y);
		float filter_value = 0.0;
		if(foreground_indicator == 0) {
			filter_value = total_value / (total_weight + 1e-3f);
		}
		

		//Write to the surface
		surf2Dwrite(foreground_indicator, upsampled_mask, x * sizeof(unsigned char), y);
		surf2Dwrite(filter_value, filter_mask, x * sizeof(float), y);
	}

	__global__ void filterForegroundMaskKernel(
		cudaTextureObject_t foreground_mask, 
		unsigned mask_rows, unsigned mask_cols,
		const float sigma,
		cudaSurfaceObject_t filter_mask
	) {
		const int x = threadIdx.x + blockDim.x * blockIdx.x;
		const int y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x >= mask_cols || y >= mask_rows) return;

		//A window search
		const int halfsize = __float2uint_ru(sigma) * 2;
		float total_weight = 0.0f;
		float total_value = 0.0f;
		for(int neighbor_y = y - halfsize; neighbor_y <= y + halfsize; neighbor_y++) {
			for(int neighbor_x = x - halfsize; neighbor_x <= x + halfsize; neighbor_x++) {
				//Retrieve the mask value at neigbour
				const unsigned char neighbor_foreground = tex2D<unsigned char>(foreground_mask, neighbor_x, neighbor_y);

				//Compute the gaussian weight
				const float diff_x_square = (neighbor_x - x) * (neighbor_x - x);
				const float diff_y_square = (neighbor_y - y) * (neighbor_y - y);
				const float weight = __expf(0.5f * (diff_x_square + diff_y_square) / (sigma * sigma));

				//Accumlate it
				if(neighbor_x >= 0 && neighbor_x < mask_cols && neighbor_y >= 0 && neighbor_y < mask_rows)
				{
					total_weight += weight;
					total_value += weight * float(1 - neighbor_foreground);
				}
			}
		}

		
		//Compute the value locally
		const unsigned char foreground_indicator = tex2D<unsigned char>(foreground_mask, x, y);
		float filter_value = 0.0;
		if(foreground_indicator == 0) {
			filter_value = total_value / (total_weight + 1e-3f);
		}
		

		//Write to the surface
		surf2Dwrite(filter_value, filter_mask, x * sizeof(float), y);
	}

} // device
} // surfelwarp

void surfelwarp::ForegroundSegmenter::UpsampleFilterForegroundMask(
	cudaTextureObject_t subsampled_mask,
	unsigned subsampled_rows, unsigned subsampled_cols,
	unsigned subsample_rate,
	float sigma,
	cudaSurfaceObject_t upsampled_mask, 
	cudaSurfaceObject_t filter_mask,
	cudaStream_t stream
) {
	//Compute the size
	const auto upsampled_rows = subsampled_rows * subsample_rate;
	const auto upsampled_cols = subsampled_cols * subsample_rate;
	dim3 blk(16, 16);
	dim3 grid(divUp(upsampled_cols, blk.x), divUp(upsampled_rows, blk.y));

	//Invoke the kernel
	device::upsampleFilterForegroundMaskKernel<<<grid, blk, 0, stream>>>(
		subsampled_mask, 
		upsampled_rows, upsampled_cols, 
		subsample_rate, 
		sigma, 
		upsampled_mask, 
		filter_mask
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::ForegroundSegmenter::FilterForegroundMask(
	cudaTextureObject_t foreground_mask,
	unsigned mask_rows, unsigned mask_cols, 
	float sigma, 
	cudaSurfaceObject_t filter_mask,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(mask_cols, blk.x), divUp(mask_rows, blk.y));
	device::filterForegroundMaskKernel<<<grid, blk, 0, stream>>>(
		foreground_mask, 
		mask_rows, mask_cols, 
		sigma, 
		filter_mask
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}