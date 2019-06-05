#include "common/ConfigParser.h"
#include "common/sanity_check.h"
#include "common/logging.h"
#include "core/warp_solver/DenseDepthHandler.h"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "DenseDepthHandler.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	__global__ void markMatchedGeometryPixelPairsKernel(
		cudaTextureObject_t depth_vertex_confid_map,
		cudaTextureObject_t depth_normal_radius_map,
		cudaTextureObject_t reference_vertex_map,
		cudaTextureObject_t reference_normal_map,
		cudaTextureObject_t index_map,
		const DeviceArrayView2D<KNNAndWeight> knn_map,
		const DualQuaternion* device_warp_field,
		const Intrinsic intrinsic, const mat34 world2camera,
		unsigned* reference_pixel_matched_indicator,
		ushort2* pixel_pairs_array
	) {
		const int x = threadIdx.x + blockDim.x*blockIdx.x;
		const int y = threadIdx.y + blockDim.y*blockIdx.y;
		if (x >= knn_map.Cols() || y >= knn_map.Rows()) return;

		//The indicator will must be written to pixel_occupied_array
		unsigned valid_indicator = 0;
		ushort2 pixel_pair = make_ushort2(0xFFFF, 0xFFFF);

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
			
			//Project the vertex into image
			const int2 img_coord = {
				__float2int_rn(((warped_vertex_camera.x / (warped_vertex_camera.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x),
				__float2int_rn(((warped_vertex_camera.y / (warped_vertex_camera.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y)
			};

			if (img_coord.x >= 0 && img_coord.x < knn_map.Cols() && img_coord.y >= 0 && img_coord.y < knn_map.Rows())
			{
				const float4 depth_vertex4 = tex2D<float4>(depth_vertex_confid_map, img_coord.x, img_coord.y);
				const float4 depth_normal4 = tex2D<float4>(depth_normal_radius_map, img_coord.x, img_coord.y);
				const float3 depth_vertex = make_float3(depth_vertex4.x, depth_vertex4.y, depth_vertex4.z);
				const float3 depth_normal = make_float3(depth_normal4.x, depth_normal4.y, depth_normal4.z);

				//Check the matched
				bool valid_pair = true;

				//The depth pixel is not valid
				if(is_zero_vertex(depth_vertex4)) {
					valid_pair = false;
				}
				
				//The orientation is not matched
				if (dot(depth_normal, warped_normal_camera) < d_correspondence_normal_dot_threshold) {
					valid_pair = false;
				}
				
				//The distance is too far away
				if (squared_norm(depth_vertex - warped_vertex_camera) > d_correspondence_distance_threshold_square) {
					valid_pair = false;
				}

				//Update if required
				if(valid_pair) {
					valid_indicator = 1;
					pixel_pair.x = img_coord.x; 
					pixel_pair.y = img_coord.y;
				}
			} // The vertex project to a valid depth pixel
		} // The reference vertex is valid

		//Write to output
		const int offset = y * knn_map.Cols() + x;
		if(valid_indicator > 0) {
			reference_pixel_matched_indicator[offset] = valid_indicator;
			pixel_pairs_array[offset] = pixel_pair;
		} 
		else {
			reference_pixel_matched_indicator[offset] = 0;
			//Do not care pixel pair
		}
	}


	__global__ void markPotentialMatchedDepthPairKernel(
		cudaTextureObject_t index_map,
		unsigned img_rows, unsigned img_cols,
		unsigned* reference_pixel_matched_indicator
	) {
		const auto x = threadIdx.x + blockDim.x*blockIdx.x;
		const auto y = threadIdx.y + blockDim.y*blockIdx.y;
		if (x >= img_cols || y >= img_rows) return;

		//The indicator will must be written to pixel_occupied_array
		const auto offset = y * img_cols + x;

		//Read the value on index map
		const auto surfel_index = tex2D<unsigned>(index_map, x, y);

		//Need other criterion?
		unsigned indicator = 0;
		if(surfel_index != d_invalid_index) {
			indicator = 1;
		}
		
		reference_pixel_matched_indicator[offset] = indicator;
	}

	
	__global__ void compactMatchedPixelPairsKernel(
		const DeviceArrayView2D<KNNAndWeight> knn_map,
		const unsigned* reference_pixel_matched_indicator,
		const unsigned* prefixsum_matched_indicator,
		const ushort2* pixel_pairs_map,
		ushort4* valid_pixel_pairs_array,
		ushort4* valid_pixel_pairs_knn,
		float4* valid_pixel_pairs_knn_weight
	) {
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x >= knn_map.Cols() || y >= knn_map.Rows()) return;

		const auto flatten_idx = x + y * knn_map.Cols();
		if(reference_pixel_matched_indicator[flatten_idx] > 0)
		{
			const auto offset = prefixsum_matched_indicator[flatten_idx] - 1;
			const KNNAndWeight knn = knn_map(y, x);
			const ushort2 target_pixel = pixel_pairs_map[flatten_idx];
			valid_pixel_pairs_array[offset] = make_ushort4(x, y, target_pixel.x, target_pixel.y);
			valid_pixel_pairs_knn[offset] = knn.knn;
			valid_pixel_pairs_knn_weight[offset] = knn.weight;
		}
	}

	__global__ void compactPontentialMatchedPixelPairsKernel(
		const DeviceArrayView2D<KNNAndWeight> knn_map,
		const unsigned* reference_pixel_matched_indicator,
		const unsigned* prefixsum_matched_indicator,
		ushort2* potential_matched_pixels,
		ushort4* potential_matched_knn,
		float4*  potential_matched_knn_weight
	) {
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x >= knn_map.Cols() || y >= knn_map.Rows()) return;
		const auto flatten_idx = x + y * knn_map.Cols();
		if(reference_pixel_matched_indicator[flatten_idx] > 0)
		{
			const auto offset = prefixsum_matched_indicator[flatten_idx] - 1;
			const KNNAndWeight knn = knn_map(y, x);
			potential_matched_pixels[offset] = make_ushort2(x, y);
			potential_matched_knn[offset] = knn.knn;
			potential_matched_knn_weight[offset] = knn.weight;
		}

	}

} // namespace device
} // namespace surfelwarp




/* The methods for mark pixel pairs
 */
void surfelwarp::DenseDepthHandler::MarkMatchedPixelPairs(cudaStream_t stream) {
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_width, blk.x), divUp(m_image_height, blk.y));
	device::markMatchedGeometryPixelPairsKernel<<<grid, blk, 0, stream>>>(
		//Maps from depth observation
		m_depth_observation.vertex_map,
		m_depth_observation.normal_map,
		//Rendered maps
		m_geometry_maps.reference_vertex_map,
		m_geometry_maps.reference_normal_map,
		m_geometry_maps.index_map,
		m_knn_map,
		m_node_se3,
		//The camera info
		m_project_intrinsic, m_world2camera,
		//The output
		m_pixel_match_indicator.ptr(), m_pixel_pair_maps.ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

/* The method to compact the matched pixel pairs
 */
void surfelwarp::DenseDepthHandler::CompactMatchedPixelPairs(cudaStream_t stream) {
	//Do a prefix sum
	m_indicator_prefixsum.InclusiveSum(m_pixel_match_indicator, stream);

	//Invoke the kernel
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_width, blk.x), divUp(m_image_height, blk.y));
	device::compactMatchedPixelPairsKernel<<<grid, blk, 0, stream>>>(
		m_knn_map,
		m_pixel_match_indicator, 
		m_indicator_prefixsum.valid_prefixsum_array, 
		m_pixel_pair_maps, 
		m_valid_pixel_pairs.Ptr(),
		m_dense_depth_knn.Ptr(),
		m_dense_depth_knn_weight.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::DenseDepthHandler::SyncQueryCompactedArraySize(cudaStream_t stream)
{
	//Sync the stream and query the size
	unsigned num_valid_pairs;
	cudaSafeCall(cudaMemcpyAsync(
		&num_valid_pairs,
		m_indicator_prefixsum.valid_prefixsum_array.ptr() + m_pixel_match_indicator.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost, stream
	));
	cudaSafeCall(cudaStreamSynchronize(stream));
	
	//Correct the size of array
	m_valid_pixel_pairs.ResizeArrayOrException(num_valid_pairs);
	m_dense_depth_knn.ResizeArrayOrException(num_valid_pairs);
	m_dense_depth_knn_weight.ResizeArrayOrException(num_valid_pairs);
}