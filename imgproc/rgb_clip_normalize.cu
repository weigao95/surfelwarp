#include "common/common_types.h"
#include "common/global_configs.h"
#include "common/encode_utils.h"
#include "common/color_transfer.h"
#include "imgproc/generate_maps.h"
#include "imgproc/rgb_clip_normalize.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__global__ void clipNormalizeRGBImageKernel(
		const PtrSz<const uchar3> raw_rgb_img,
		const unsigned clip_rows, const unsigned clip_cols,
		cudaSurfaceObject_t clip_rgb_img
	) {
		//Check the position of this kernel
		const auto clip_x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto clip_y = threadIdx.y + blockDim.y * blockIdx.y;
		if (clip_x >= clip_cols || clip_y >= clip_rows) return;

		//From here, the access to raw_rgb_img should be in range
		const auto raw_x = clip_x + boundary_clip;
		const auto raw_y = clip_y + boundary_clip;
		const auto raw_flatten = raw_x + raw_y * (clip_cols + 2 * boundary_clip);
		const uchar3 raw_pixel = raw_rgb_img[raw_flatten];

		//Normalize and write to output
		float4 noramlized_rgb;
		noramlized_rgb.x = float(raw_pixel.x) / 255.0f;
		noramlized_rgb.y = float(raw_pixel.y) / 255.0f;
		noramlized_rgb.z = float(raw_pixel.z) / 255.0f;
		noramlized_rgb.w = 1.0f;
		surf2Dwrite(noramlized_rgb, clip_rgb_img, clip_x * sizeof(float4), clip_y);
	}


	__global__ void clipNormalizeRGBImageKernel(
		const PtrSz<const uchar3> raw_rgb_img,
		const unsigned clip_rows, const unsigned clip_cols,
		cudaSurfaceObject_t clip_rgb_img,
		cudaSurfaceObject_t density_map
	) {
		//Check the position of this kernel
		const auto clip_x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto clip_y = threadIdx.y + blockDim.y * blockIdx.y;
		if (clip_x >= clip_cols || clip_y >= clip_rows) return;

		//From here, the access to raw_rgb_img should be in range
		const auto raw_x = clip_x + boundary_clip;
		const auto raw_y = clip_y + boundary_clip;
		const auto raw_flatten = raw_x + raw_y * (clip_cols + 2 * boundary_clip);
		const uchar3 raw_pixel = raw_rgb_img[raw_flatten];

		//Normalize and write to output
		float4 noramlized_rgb;
		noramlized_rgb.x = float(raw_pixel.x) / 255.0f;
		noramlized_rgb.y = float(raw_pixel.y) / 255.0f;
		noramlized_rgb.z = float(raw_pixel.z) / 255.0f;
		noramlized_rgb.w = 1.0f;
		const float density = rgba2density(noramlized_rgb);

		surf2Dwrite(noramlized_rgb, clip_rgb_img, clip_x * sizeof(float4), clip_y);
		surf2Dwrite(density, density_map, clip_x * sizeof(float), clip_y);
	}


	__global__ void createColorTimeMapKernel(
		const PtrSz<const uchar3> raw_rgb_img,
		const unsigned clip_rows, const unsigned clip_cols,
		const float init_time,
		cudaSurfaceObject_t color_time_map
	){
		const auto clip_x = threadIdx.x + blockIdx.x * blockDim.x;
		const auto clip_y = threadIdx.y + blockIdx.y * blockDim.y;
		if(clip_x >= clip_cols || clip_y >= clip_rows) return;
		
		//From here, the access to raw_rgb_img should be in range
		const auto raw_x = clip_x + boundary_clip;
		const auto raw_y = clip_y + boundary_clip;
		const auto raw_flatten = raw_x + raw_y * (clip_cols + 2 * boundary_clip);
		const uchar3 raw_pixel = raw_rgb_img[raw_flatten];
		const float encoded_pixel = float_encode_rgb(raw_pixel);

		//Construct the result and store it
		const float4 color_time_value = make_float4(encoded_pixel, 0, init_time, init_time);
		surf2Dwrite(color_time_value, color_time_map, clip_x * sizeof(float4), clip_y);
	}


	__global__ void filterDensityMapKernel(
		cudaTextureObject_t density_map,
		unsigned rows, unsigned cols,
		cudaSurfaceObject_t filter_density_map
	) {
		const auto x = threadIdx.x + blockIdx.x * blockDim.x;
		const auto y = threadIdx.y + blockIdx.y * blockDim.y;
		if(x >= cols || y >= rows) return;

		const auto half_width = 5;
		const float center_density = tex2D<float>(density_map, x, y);

		//The window search
		float sum_all = 0.0f; float sum_weight = 0.0f;
		for(auto y_idx = y - half_width; y_idx <= y + half_width; y_idx++) {
			for(auto x_idx = x - half_width; x_idx <= x + half_width; x_idx++) {
				const float density = tex2D<float>(density_map, x_idx, y_idx);
				const float value_diff2 = (center_density - density) * (center_density - density);
				const float pixel_diff2 = (x_idx - x) * (x_idx - x) + (y_idx - y) * (y_idx - y);
				const float this_weight = (density > 0.0f) * expf(-(1.0f / 25) * pixel_diff2) * expf(-(1.0f / 0.01) * value_diff2);
				sum_weight += this_weight;
				sum_all += this_weight * density;
			}
		}

		//The filter value
		float filter_density_value = sum_all / (sum_weight);
		
		//Clip the value to suitable range
		if(filter_density_value >= 1.0f) {
			filter_density_value = 1.0f;
		} else if(filter_density_value >= 0.0f) {
			//pass
		} else {
			filter_density_value = 0.0f;
		}
		//if(isnan(filter_density_value)) printf("Nan in the image");
		surf2Dwrite(filter_density_value, filter_density_map, x * sizeof(float), y);
	}
}; /* End of namespace device */
}; /* End of namespace surfelwarp */

void surfelwarp::clipNormalizeRGBImage(
	const DeviceArray<uchar3> raw_rgb_img, 
	const unsigned clip_rows, const unsigned clip_cols, 
	cudaSurfaceObject_t clip_rgb_img, 
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(clip_cols, blk.x), divUp(clip_rows, blk.y));
	device::clipNormalizeRGBImageKernel<<<grid, blk, 0, stream>>>(
		raw_rgb_img, 
		clip_rows, clip_cols, 
		clip_rgb_img
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::clipNormalizeRGBImage(
	const DeviceArray<uchar3>& raw_rgb_img, 
	unsigned clip_rows, unsigned clip_cols,
	cudaSurfaceObject_t clip_rgb_img, 
	cudaSurfaceObject_t density_map, 
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(clip_cols, blk.x), divUp(clip_rows, blk.y));
	device::clipNormalizeRGBImageKernel<<<grid, blk, 0, stream>>>(
		raw_rgb_img,
		clip_rows, clip_cols, 
		clip_rgb_img, 
		density_map
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::createColorTimeMap(
	const DeviceArray<uchar3> raw_rgb_img, 
	const unsigned clip_rows, const unsigned clip_cols,
	const float init_time,
	cudaSurfaceObject_t color_time_map, 
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(clip_cols, blk.x), divUp(clip_rows, blk.y));
	device::createColorTimeMapKernel<<<grid, blk, 0, stream>>>(
		raw_rgb_img, 
		clip_rows, clip_cols, 
		init_time, 
		color_time_map
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::filterDensityMap(
	cudaTextureObject_t density_map,
	cudaSurfaceObject_t filter_density_map,
	unsigned rows, unsigned cols,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(cols, blk.x), divUp(rows, blk.y));
	device::filterDensityMapKernel<<<grid, blk, 0, stream>>>(
		density_map,
		rows, cols,
		filter_density_map
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}