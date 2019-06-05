#include "common/common_types.h"
#include "common/Constants.h"
#include "imgproc/depth_clip_filter.h"
#include "math/device_mat.h"
#include "math/vector_ops.hpp"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	__global__ void clipFilterDepthKernel(
		cudaTextureObject_t raw_depth,
		const unsigned clip_img_rows,
		const unsigned clip_img_cols,
		const unsigned clip_near,
		const unsigned clip_far,
		const float sigma_s_inv_square, 
		const float sigma_r_inv_square,
		cudaSurfaceObject_t filter_depth
	) {
		//Parallel over the clipped image
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (y >= clip_img_rows || x >= clip_img_cols) return;

		//Compute the center on raw depth
		const auto half_width = 5;
		const auto raw_x = x + boundary_clip;
		const auto raw_y = y + boundary_clip;
		const unsigned short center_depth = tex2D<unsigned short>(raw_depth, raw_x, raw_y);

		//Iterate over the window
		float sum_all = 0.0f; float sum_weight = 0.0f;
		for(auto y_idx = raw_y - half_width; y_idx <= raw_y + half_width; y_idx++) {
			for(auto x_idx = raw_x - half_width; x_idx <= raw_x + half_width; x_idx++) {
				const unsigned short depth = tex2D<unsigned short>(raw_depth, x_idx, y_idx);
				const float depth_diff2 = (depth - center_depth) * (depth - center_depth);
				const float pixel_diff2 = (x_idx - raw_x) * (x_idx - raw_x) + (y_idx - raw_y) * (y_idx - raw_y);
				const float this_weight = (depth > 0) * expf(-sigma_s_inv_square * pixel_diff2) * expf(-sigma_r_inv_square * depth_diff2);
				sum_weight += this_weight;
				sum_all += this_weight * depth;
			}
		}

		//Put back to the filtered depth
		unsigned short filtered_depth_value = __float2uint_rn(sum_all / sum_weight);
		if (filtered_depth_value < clip_near || filtered_depth_value > clip_far) filtered_depth_value = 0;
		surf2Dwrite(filtered_depth_value, filter_depth, x * sizeof(unsigned short), y);
	}


	__global__ void reprojectDepthToRGBKernel(
		cudaTextureObject_t depth_map,
		const unsigned rows, const unsigned cols,
		const IntrinsicInverse depth_intrinsic_inv, 
		const Intrinsic raw_rgb_intrinsic,
		const mat34 depth2rgb,
		const int reproject_scale,
		PtrStepSz<unsigned short> reprojected_depth
	) {
		//Obtain the index
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x >= cols || y >= rows) return;

		//Read the original depth
		const unsigned short depth_value = tex2D<unsigned short>(depth_map, x, y);

		//Back project into 3d space
		float3 coord;
		coord.z = float(depth_value) * 0.001f; //into [mm]
		coord.x = (x - depth_intrinsic_inv.principal_x) * depth_intrinsic_inv.inv_focal_x * coord.z;
		coord.y = (y - depth_intrinsic_inv.principal_y) * depth_intrinsic_inv.inv_focal_y * coord.z;

		//Transform into the rgb frame
		coord = depth2rgb.rot * coord + depth2rgb.trans;

		//Project it into rgb
		int2 reproject_img_coord = {
			__float2int_rn(((coord.x / (coord.z + 1e-10)) * raw_rgb_intrinsic.focal_x) + raw_rgb_intrinsic.principal_x),
			__float2int_rn(((coord.y / (coord.z + 1e-10)) * raw_rgb_intrinsic.focal_y) + raw_rgb_intrinsic.principal_y)
		};

		//Scale it
		reproject_img_coord.x *= reproject_scale;
		reproject_img_coord.y *= reproject_scale;

		//In side the region, the store it
		if(reproject_img_coord.x >= 0 
			&& reproject_img_coord.x < reprojected_depth.cols
			&& reproject_img_coord.y >= 0 
			&& reproject_img_coord.y < reprojected_depth.rows)
		{
			const unsigned short depth_new = __float2uint_rn((coord.z * 1000));
			reprojected_depth.ptr(reproject_img_coord.y)[reproject_img_coord.x] = depth_new;
		}
	}

	__global__ void collectReprojectedDepthKernel(
		const PtrStepSz<const unsigned short> reprojected_depth,
		const int reproject_scale,
		const unsigned raw_cols, const unsigned raw_rows,
		cudaSurfaceObject_t reproject_raw_depth
	) {
		//The index on the raw map
		const auto raw_x = threadIdx.x + blockIdx.x * blockDim.x;
		const auto raw_y = threadIdx.y + blockIdx.y * blockDim.y;
		if (raw_x >= raw_cols || raw_y >= raw_rows) return;
		
		//The index on the reprojected map
		const auto reproject_x = reproject_scale * raw_x;
		const auto reproject_y = reproject_scale * raw_y;

		//Collect it
		float depth_sum = 0.0f;
		int valid_count = 0;
		for(auto y = reproject_y; y < reproject_y + reproject_scale; y++) {
			for(auto x = reproject_x; x < reproject_x + reproject_scale; x++) {
				//Direct access, not memory violation
				const auto depth = reprojected_depth.ptr(y)[x];
				if(depth > 0) {
					depth_sum += float(depth);
					valid_count++;
				}
			}
		}

		unsigned short depth_value = 0;
		if(valid_count > 0) {
			depth_value = __float2uint_rn((depth_sum / valid_count));
		}
		surf2Dwrite(depth_value, reproject_raw_depth, raw_x * sizeof(unsigned short), raw_y);
	}

};/* End of namespace device */
};/* End of namespace surfelwarp */






void surfelwarp::clipFilterDepthImage(
	cudaTextureObject_t raw_depth, 
	const unsigned clip_img_rows, const unsigned clip_img_cols,
	const unsigned clip_near, const unsigned clip_far, 
	cudaSurfaceObject_t filter_depth,
	cudaStream_t stream
) {
	//The filter parameters
	const float sigma_s = Constants::kFilterSigma_S;
	const float sigma_r = Constants::kFilterSigma_R;
	const float sigma_s_inv_square = 1.0f / (sigma_s * sigma_s);
	const float sigma_r_inv_square = 1.0f / (sigma_r * sigma_r);

	//Invoke the kernel
	dim3 blk(16, 16);
	dim3 grid(divUp(clip_img_cols, blk.x), divUp(clip_img_rows, blk.y));
	device::clipFilterDepthKernel<<<grid, blk, 0, stream>>>(
		raw_depth, 
		clip_img_rows, clip_img_cols, 
		clip_near, clip_far, 
		sigma_s_inv_square, sigma_r_inv_square, 
		filter_depth
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}



void surfelwarp::reprojectDepthToRGB(
	cudaTextureObject_t raw_depth, 
	cudaSurfaceObject_t reprojected_depth_surface,
	DeviceArray2D<unsigned short> reprojected_buffer,
	const unsigned raw_rows, const unsigned raw_cols, 
	const IntrinsicInverse & raw_depth_intrinsic_inverse, 
	const Intrinsic & raw_rgb_intrinsic, 
	const mat34 & depth2rgb,
	cudaStream_t stream
) {
	//Invoke the reproject kernel
	const auto factor = Constants::kReprojectScaleFactor;
	dim3 blk(16, 16);
	dim3 grid(divUp(raw_cols, blk.x), divUp(raw_rows, blk.y));
	device::reprojectDepthToRGBKernel<<<grid, blk, 0, stream>>>(
		raw_depth, 
		raw_rows, raw_cols,
		raw_depth_intrinsic_inverse, 
		raw_rgb_intrinsic, 
		depth2rgb, 
		factor,
		reprojected_buffer
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	//Invoke the collect kernel
	device::collectReprojectedDepthKernel<<<grid, blk, 0, stream>>>(
		reprojected_buffer, 
		factor, 
		raw_cols, raw_rows, 
		reprojected_depth_surface
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}
