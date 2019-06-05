#include "common/common_types.h"
#include "imgproc/image_gradient.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	

	/**
	 * \brief The format is
	 *        v0, v3, v5
	 *        v1,   , v6
	 *        v2, v4, v7
	 */
	__host__ __device__ void computeImageGradient(
		const float v[8],
		float& dv_dx, float& dv_dy
	){
		dv_dx = 0.0625f * (-3 * v[0] + 3 * v[5] - 10 * v[1] + 10 * v[6] - 3 * v[2] + 3 * v[7]);
		dv_dy = 0.0625f * (-3 * v[0] + 3 * v[2] - 10 * v[3] + 10 * v[4] - 3 * v[5] + 3 * v[7]);
	}

	__global__ void computeDensityForegroundMaskGradientKernel(
		cudaTextureObject_t filter_foreground_mask,
		cudaTextureObject_t density_map,
		unsigned rows, unsigned cols,
		cudaSurfaceObject_t foreground_mask_gradient_map,
		cudaSurfaceObject_t density_gradient_map
	) {
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x >= cols || y >= rows) return;

		//Now the gradient must be written to surface
		float map_value[8];
		float2 mask_gradient, density_gradient;

		//Load it and compute
		map_value[0] = tex2D<float>(filter_foreground_mask, x-1, y-1);
		map_value[1] = tex2D<float>(filter_foreground_mask, x-1, y  );
		map_value[2] = tex2D<float>(filter_foreground_mask, x-1, y+1);
		map_value[3] = tex2D<float>(filter_foreground_mask, x  , y-1);
		map_value[4] = tex2D<float>(filter_foreground_mask, x  , y+1);
		map_value[5] = tex2D<float>(filter_foreground_mask, x+1, y-1);
		map_value[6] = tex2D<float>(filter_foreground_mask, x+1, y  );
		map_value[7] = tex2D<float>(filter_foreground_mask, x+1, y+1);
		computeImageGradient(map_value, mask_gradient.x, mask_gradient.y);

		map_value[0] = tex2D<float>(density_map, x-1, y-1);
		map_value[1] = tex2D<float>(density_map, x-1, y  );
		map_value[2] = tex2D<float>(density_map, x-1, y+1);
		map_value[3] = tex2D<float>(density_map, x  , y-1);
		map_value[4] = tex2D<float>(density_map, x  , y+1);
		map_value[5] = tex2D<float>(density_map, x+1, y-1);
		map_value[6] = tex2D<float>(density_map, x+1, y  );
		map_value[7] = tex2D<float>(density_map, x+1, y+1);
		computeImageGradient(map_value, density_gradient.x, density_gradient.y);

		//Store the value to surface
		surf2Dwrite(mask_gradient, foreground_mask_gradient_map, x * sizeof(float2), y);
		surf2Dwrite(density_gradient, density_gradient_map, x * sizeof(float2), y);
	}

} // namespace device
} // namespace surfelwarp


void surfelwarp::computeDensityForegroundMaskGradient(
	cudaTextureObject_t filter_foreground_mask,
	cudaTextureObject_t density_map, 
	unsigned rows, unsigned cols, 
	cudaSurfaceObject_t foreground_mask_gradient_map,
	cudaSurfaceObject_t density_gradient_map,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(cols, blk.x), divUp(rows, blk.y));
	device::computeDensityForegroundMaskGradientKernel<<<grid, blk, 0, stream>>>(
		filter_foreground_mask, 
		density_map, 
		rows, cols, 
		foreground_mask_gradient_map,
		density_gradient_map
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

