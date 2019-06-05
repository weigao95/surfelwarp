#include "common/common_types.h"
#include "imgproc/depth_surfel_collect.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__global__ void markValicdDepthPixelKernel(
		cudaTextureObject_t depth_img,
		const unsigned rows, const unsigned cols,
		PtrSz<char> valid_indicator
	) {
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x >= cols || y >= rows) return;

		//This indicator must finally written to output array
		const auto flatten_idx = x + cols * y;
		char valid = 0;
		const unsigned short depth_value = tex2D<unsigned short>(depth_img, x, y);
		if(depth_value > 0) {
			valid = 1;
		}

		//Write to output
		valid_indicator[flatten_idx] = valid;
	}


	__global__ void collectDepthSurfelKernel(
		cudaTextureObject_t vertex_confid_map,
		cudaTextureObject_t normal_radius_map,
		cudaTextureObject_t color_time_map,
		const PtrSz<const int> selected_array,
		const unsigned rows, const unsigned cols,
		PtrSz<DepthSurfel> valid_depth_surfel
	) {
		const auto selected_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (selected_idx >= selected_array.size) return;
		const auto idx = selected_array[selected_idx];
		const auto x = idx % cols;
		const auto y = idx / cols;
		
		//Construct the output
		DepthSurfel surfel;
		surfel.pixel_coord.x() = x;
		surfel.pixel_coord.y() = y;
		surfel.vertex_confid = tex2D<float4>(vertex_confid_map, x, y);
		surfel.normal_radius = tex2D<float4>(normal_radius_map, x, y);
		surfel.color_time = tex2D<float4>(color_time_map, x, y);
		
		//Write to the output array
		valid_depth_surfel[selected_idx] = surfel;
	}

}; /* End of namespace deivce */
}; /* End of namespace surfelwarp */

void surfelwarp::markValidDepthPixel(
	cudaTextureObject_t depth_img,
	const unsigned rows, const unsigned cols,
	DeviceArray<char>& valid_indicator,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(cols, blk.x), divUp(rows, blk.y));
	device::markValicdDepthPixelKernel<<<grid, blk, 0, stream>>>(
		depth_img,
		rows, cols,
		valid_indicator
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::collectDepthSurfel(
	cudaTextureObject_t vertex_confid_map, 
	cudaTextureObject_t normal_radius_map,
	cudaTextureObject_t color_time_map,
	const DeviceArray<int>& selected_array,
	const unsigned rows, const unsigned cols,
	DeviceArray<DepthSurfel>& valid_depth_surfel,
	cudaStream_t stream
) {
	dim3 blk(128);
	dim3 grid(divUp(selected_array.size(), blk.x));
	device::collectDepthSurfelKernel<<<grid, blk, 0, stream>>>(
		vertex_confid_map,
		normal_radius_map,
		color_time_map,
		selected_array,
		rows, cols,
		valid_depth_surfel
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}