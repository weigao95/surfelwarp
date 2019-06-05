#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/surfel_types.h"
#include "common/ConfigParser.h"
#include "common/sanity_check.h"
#include "math/device_mat.h"
#include "math/DualQuaternion.hpp"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/DensityForegroundMapHandler.h"
#include "core/warp_solver/density_map_jacobian.cuh"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	enum {
		valid_color_halfsize = 1
	};

	__device__ __forceinline__ bool isValidColorPixel(
		cudaTextureObject_t rendered_rgb_map,
		cudaTextureObject_t index_map,
		int x_center, int y_center
	){
#if defined(USE_DENSE_SOLVER_MAPS)
		for(auto y = y_center - valid_color_halfsize; y <= y_center + valid_color_halfsize; y++) {
			for(auto x = x_center - valid_color_halfsize; x <= x_center + valid_color_halfsize; x++) {
				if(tex2D<unsigned>(index_map, x, y) == d_invalid_index) return false;
			}
		}
#endif
		return true;
	}

	__device__ __forceinline__ bool isValidForegroundMaskPixel(
		cudaTextureObject_t foreground_mask,
		int x, int y
	) {
		const auto mask_value = tex2D<float>(foreground_mask, x, y);
		// Note that the mask is INVERSED, 0 means foreground
		return (mask_value > 1e-5f);
	}

	__global__ void markValidColorForegroundMapsPixelKernel(
		cudaTextureObject_t rendered_rgb_map,
		cudaTextureObject_t reference_vertex_map,
		cudaTextureObject_t reference_normal_map,
		cudaTextureObject_t index_map,
		cudaTextureObject_t filtered_foreground_mask,
		const DeviceArrayView2D<KNNAndWeight> knn_map,
		const DualQuaternion* device_warp_field,
		const Intrinsic intrinsic, const mat34 world2camera,
		unsigned* valid_rgb_indicator_array,
		unsigned* valid_foregound_mask_indicator_array
	) {
		const int x = threadIdx.x + blockDim.x*blockIdx.x;
		const int y = threadIdx.y + blockDim.y*blockIdx.y;
		if (x >= knn_map.Cols() || y >= knn_map.Rows()) return;

		//The indicator will must be written to pixel_occupied_array
		unsigned valid_rgb = 0;
		unsigned valid_foreground_mask = 0;

		//Read the value on index map
		const auto surfel_index = tex2D<unsigned>(index_map, x, y);
		if(surfel_index != d_invalid_index)
		{
			//Get the vertex
			const float4 can_vertex4 = tex2D<float4>(reference_vertex_map, x, y);
			const float4 can_normal4 = tex2D<float4>(reference_normal_map, x, y);
			const KNNAndWeight knn = knn_map(y, x);
			DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn.knn, knn.weight);
			const mat34 se3 = dq_average.se3_matrix();

			//And warp it
			const float3 warped_vertex = se3.rot * can_vertex4 + se3.trans;
			const float3 warped_normal = se3.rot * can_normal4;
			
			//Transfer to the camera frame
			const float3 warped_vertex_camera = world2camera.rot * warped_vertex + world2camera.trans;
			const float3 warped_normal_camera = world2camera.rot * warped_normal;
			//Check the normal of this pixel
			const float3 view_direction = normalized(warped_vertex_camera);
			const float viewangle_cos = - dot(view_direction, warped_normal_camera);
			
			//Project the vertex into image
			//The image coordinate of this vertex
			const int2 img_coord = {
				__float2int_rn(((warped_vertex_camera.x / (warped_vertex_camera.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x),
				__float2int_rn(((warped_vertex_camera.y / (warped_vertex_camera.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y)
			};

			//Check that the pixel projected to and the view angle
			if (img_coord.x >= 0 && img_coord.x < knn_map.Cols()
				&& img_coord.y >= 0 && img_coord.y < knn_map.Rows() 
				&& viewangle_cos > d_valid_color_dot_threshold
				)
			{
				if(isValidColorPixel(rendered_rgb_map, index_map, x, y)) valid_rgb = 1;
				if(isValidForegroundMaskPixel(filtered_foreground_mask, img_coord.x, img_coord.y)) valid_foreground_mask = 1;
			} // The vertex project to a valid image pixel

			//Mark the rgb as always valid if the pixel is valid?
			valid_rgb = 1;

		} // The reference vertex is valid

		//Write to output
		const auto offset = y * knn_map.Cols() + x;
		valid_rgb_indicator_array[offset] = valid_rgb;
		//valid_rgb_indicator_array[offset] = 0; //Check whether the pipeline work without density term
		valid_foregound_mask_indicator_array[offset] = valid_foreground_mask;
	}
	


	__global__ void compactValidPixelKernel(
		const unsigned* valid_indicator_array,
		const unsigned* prefixsum_indicator_array,
		const DeviceArrayView2D<KNNAndWeight> knn_map,
		ushort2* compacted_pixel_coordinate,
		ushort4* pixel_knn,
		float4* pixel_knn_weight
	) {
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x >= knn_map.Cols() || y >= knn_map.Rows()) return;
		
		const auto flatten_idx = x + y * knn_map.Cols();
		if(valid_indicator_array[flatten_idx] > 0) {
			const auto offset = prefixsum_indicator_array[flatten_idx] - 1;
			const auto knn = knn_map(y, x);
			compacted_pixel_coordinate[offset] = make_ushort2(x, y);
			pixel_knn[offset] = knn.knn;
			pixel_knn_weight[offset] = knn.weight;
		}
	}

} // namespace device
} // namespace surfelwarp



/* The method to mark the valid pixels
 */
void surfelwarp::DensityForegroundMapHandler::MarkValidColorForegroundMaskPixels(
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_width, blk.x), divUp(m_image_height, blk.y));
	device::markValidColorForegroundMapsPixelKernel<<<grid, blk, 0, stream>>>(
		//The rendered geometry maps
		m_geometry_maps.normalized_rgb_map,
		m_geometry_maps.reference_vertex_map,
		m_geometry_maps.reference_normal_map,
		m_geometry_maps.index_map,
		//From the depth observation
		m_depth_observation.filtered_foreground_mask,
		//The knn and warp field
		m_knn_map, m_node_se3,
		m_project_intrinsic, m_world2camera,
		//Output
		m_color_pixel_indicator_map.ptr(),
		m_mask_pixel_indicator_map.ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
	//LOG(INFO) << "The number of valid rgb pixels " << numNonZeroElement(m_color_pixel_indicator_map);
#endif
}


/* The method for compaction
 */
void surfelwarp::DensityForegroundMapHandler::CompactValidColorPixel(cudaStream_t stream) {
	//Do a prefix sum
	m_color_pixel_indicator_prefixsum.InclusiveSum(m_color_pixel_indicator_map, stream);
	
	//Invoke the kernel
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_width, blk.x), divUp(m_image_height, blk.y));
	device::compactValidPixelKernel<<<grid, blk, 0, stream>>>(
		m_color_pixel_indicator_map,
		m_color_pixel_indicator_prefixsum.valid_prefixsum_array,
		m_knn_map,
		//The output
		m_valid_color_pixel.Ptr(),
		m_valid_color_pixel_knn.Ptr(),
		m_valid_color_pixel_knn_weight.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::DensityForegroundMapHandler::QueryCompactedColorPixelArraySize(cudaStream_t stream)
{
	//Sync the stream and query the size
	unsigned num_valid_pairs;
	cudaSafeCall(cudaMemcpyAsync(
		&num_valid_pairs,
		m_color_pixel_indicator_prefixsum.valid_prefixsum_array.ptr() + m_color_pixel_indicator_map.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost, stream
	));
	cudaSafeCall(cudaStreamSynchronize(stream));
	
	//Correct the size of the array
	m_valid_color_pixel.ResizeArrayOrException(num_valid_pairs);
	m_valid_color_pixel_knn.ResizeArrayOrException(num_valid_pairs);
	m_valid_color_pixel_knn_weight.ResizeArrayOrException(num_valid_pairs);

	//Also write to potential pixels
	m_potential_pixels_knn.pixels = m_valid_color_pixel.ArrayReadOnly();
	m_potential_pixels_knn.node_knn = m_valid_color_pixel_knn.ArrayReadOnly();
	m_potential_pixels_knn.knn_weight = m_valid_color_pixel_knn_weight.ArrayReadOnly();
	
	//Check the output
	//LOG(INFO) << "The number of valid color pixel is " << m_valid_color_pixel.ArraySize();
}


void surfelwarp::DensityForegroundMapHandler::CompactValidMaskPixel(cudaStream_t stream) {
	//Do a prefix sum
	m_mask_pixel_indicator_prefixsum.InclusiveSum(m_mask_pixel_indicator_map, stream);
	
	//Invoke the kernel
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_width, blk.x), divUp(m_image_height, blk.y));
	device::compactValidPixelKernel<<<grid, blk, 0, stream>>>(
		m_mask_pixel_indicator_map,
		m_mask_pixel_indicator_prefixsum.valid_prefixsum_array,
		m_knn_map,
		//The output
		m_valid_mask_pixel.Ptr(),
		m_valid_mask_pixel_knn.Ptr(),
		m_valid_mask_pixel_knn_weight.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::DensityForegroundMapHandler::QueryCompactedMaskPixelArraySize(cudaStream_t stream)
{
	//Sync the stream and query the size
	cudaSafeCall(cudaMemcpyAsync(
		m_num_mask_pixel,
		m_mask_pixel_indicator_prefixsum.valid_prefixsum_array.ptr() + m_mask_pixel_indicator_map.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost, stream
	));
	cudaSafeCall(cudaStreamSynchronize(stream));
	
	//Correct the size of the array
	m_valid_mask_pixel.ResizeArrayOrException(*m_num_mask_pixel);
	m_valid_mask_pixel_knn.ResizeArrayOrException(*m_num_mask_pixel);
	m_valid_mask_pixel_knn_weight.ResizeArrayOrException(*m_num_mask_pixel);
}








