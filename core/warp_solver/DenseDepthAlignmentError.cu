#include "common/ConfigParser.h"
#include "common/sanity_check.h"
#include "common/logging.h"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/DenseDepthHandler.h"
#include "DenseDepthHandler.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	enum {
		window_halfsize = 1,
	};

	__device__ __forceinline__ float computeAlignmentErrorWindowSearch(
		cudaTextureObject_t depth_vertex_confid_map,
		cudaTextureObject_t depth_normal_radius_map,
		cudaTextureObject_t filter_foreground_mask,
		cudaTextureObject_t reference_vertex_map,
		cudaTextureObject_t reference_normal_map,
		cudaTextureObject_t index_map,
		const DeviceArrayView2D<KNNAndWeight> knn_map,
		const DualQuaternion* device_warp_field,
		const Intrinsic& intrinsic, const mat34& world2camera
	) {
		const auto x = threadIdx.x + blockDim.x*blockIdx.x;
		const auto y = threadIdx.y + blockDim.y*blockIdx.y;
		if (x >= knn_map.Cols() || y >= knn_map.Rows()) return 0.0f;

		//The residual value
		float alignment_error = 0.0f;
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

			//Use window search
			alignment_error = d_maximum_alignment_error;
			bool depth_vertex_found = false;
			for(auto depth_y = img_coord.y - window_halfsize; depth_y <= img_coord.y + window_halfsize; depth_y++) {
				for(auto depth_x = img_coord.x - window_halfsize; depth_x <= img_coord.x + window_halfsize; depth_x++) {
					const float4 depth_vertex = tex2D<float4>(depth_vertex_confid_map, depth_x, depth_y);
					const float4 depth_normal = tex2D<float4>(depth_normal_radius_map, depth_x, depth_y);
					if(!is_zero_vertex(depth_vertex) && dotxyz(warped_normal_camera, depth_normal) > 0.3f)
						depth_vertex_found = true;
					const auto error = fabsf_diff_xyz(warped_vertex_camera, depth_vertex);
					if(error < alignment_error)
						alignment_error = error;
				}
			}
			
			//If there is no depth pixel, check the foreground mask
			if(!depth_vertex_found) {
				const float filter_foreground_value = tex2D<float>(filter_foreground_mask, img_coord.x, img_coord.y);
				if(filter_foreground_value < 0.9) { //This is on boundary or foreground
					//0.05[m] (5 cm) is the approximate maximum value (corresponded to 1.0 foreground value)
					//if the surfel is on the boundary of the image.
					alignment_error = 0.03f * filter_foreground_value;
				}
			}
		}

		//Return the value for further processing
		return alignment_error;
	}

	__global__ void computeAlignmentErrorMapKernel(
		cudaTextureObject_t depth_vertex_confid_map,
		cudaTextureObject_t depth_normal_radius_map,
		cudaTextureObject_t filter_foreground_mask,
		cudaTextureObject_t reference_vertex_map,
		cudaTextureObject_t reference_normal_map,
		cudaTextureObject_t index_map,
		const DeviceArrayView2D<KNNAndWeight> knn_map,
		const DualQuaternion* device_warp_field,
		const Intrinsic intrinsic, const mat34 world2camera,
		//the output
		cudaSurfaceObject_t alignment_error_map
	) {
		const auto x = threadIdx.x + blockDim.x*blockIdx.x;
		const auto y = threadIdx.y + blockDim.y*blockIdx.y;
		if (x >= knn_map.Cols() || y >= knn_map.Rows()) return;

		//The residual value
		const float alignment_error = computeAlignmentErrorWindowSearch(
			depth_vertex_confid_map,
			depth_normal_radius_map,
			filter_foreground_mask,
			reference_vertex_map,
			reference_normal_map,
			index_map, 
			knn_map, 
			device_warp_field, 
			intrinsic, world2camera
		);

		//Write the value to surface
		surf2Dwrite(alignment_error, alignment_error_map, x * sizeof(float), y);
	}


	__global__ void computeNodeAlignmentErrorFromMapKernel(
		cudaTextureObject_t depth_vertex_confid_map,
		cudaTextureObject_t depth_normal_radius_map,
		cudaTextureObject_t filter_foreground_mask,
		cudaTextureObject_t reference_vertex_map,
		cudaTextureObject_t reference_normal_map,
		cudaTextureObject_t index_map,
		const DeviceArrayView2D<KNNAndWeight> knn_map,
		const DualQuaternion* device_warp_field,
		const Intrinsic intrinsic, const mat34 world2camera,
		//the output
		float* node_alignment_error,
		float* node_accumlate_weight
	) {
		const auto x = threadIdx.x + blockDim.x*blockIdx.x;
		const auto y = threadIdx.y + blockDim.y*blockIdx.y;
		if (x >= knn_map.Cols() || y >= knn_map.Rows()) return;

		//The residual value
		const float alignment_error = computeAlignmentErrorWindowSearch(
			depth_vertex_confid_map,
			depth_normal_radius_map,
			filter_foreground_mask,
			reference_vertex_map,
			reference_normal_map,
			index_map, 
			knn_map, 
			device_warp_field, 
			intrinsic, world2camera
		);

		//The knn and weight is used to interplate
		const KNNAndWeight knn = knn_map(y, x);
		const unsigned short* node_array = (const unsigned short*)(&knn.knn);
		const float* node_weight_array = (const float*)(&knn.weight);
		if(alignment_error > 1e-6f)
		{
			for(auto i = 0; i < 4; i++) {
				const auto node = node_array[i];
				const auto node_weight = node_weight_array[i];
				atomicAdd(&(node_alignment_error[node]), node_weight * alignment_error);
				atomicAdd(&(node_accumlate_weight[node]), node_weight);
			}
		}
	}

	__global__ void collectAlignmentErrorMapFromNodeKernel(
		const float* node_alignment_error,
		const float* node_accumlate_weight,
		cudaTextureObject_t index_map,
		const DeviceArrayView2D<KNNAndWeight> knn_map,
		//the output
		cudaSurfaceObject_t alignment_error_map
	) {
		const auto x = threadIdx.x + blockDim.x*blockIdx.x;
		const auto y = threadIdx.y + blockDim.y*blockIdx.y;
		if (x >= knn_map.Cols() || y >= knn_map.Rows()) return;

		float filter_alignment_error = 0.0f;
		const auto surfel_index = tex2D<unsigned>(index_map, x, y);
		if(surfel_index != d_invalid_index)
		{
			//The knn and weight is used to interplate
			const KNNAndWeight knn = knn_map(y, x);
			const unsigned short* node_array = (const unsigned short*)(&knn.knn);
			const float* node_weight_array = (const float*)(&knn.weight);

			//Load from node
			float accumlate_error = 0.0f;
			float accumlate_weight = 0.0f;
			for(auto i = 0; i < 4; i++) {
				const auto node = node_array[i];
				const auto node_weight = node_weight_array[i];
				const float node_error = node_alignment_error[node];
				const float node_total_weight = node_accumlate_weight[node];
				const float node_unit_error = node_error / node_total_weight;
				accumlate_error += node_unit_error * node_weight;
				accumlate_weight += node_weight;
			}

			//Write to output value
			filter_alignment_error = accumlate_error / (accumlate_weight + 1e-4f);
		}

		//Write the value to surface
		surf2Dwrite(filter_alignment_error, alignment_error_map, x * sizeof(float), y);
	}

} // device
} // surfelwarp


void surfelwarp::DenseDepthHandler::ComputeAlignmentErrorMapDirect(
	const DeviceArrayView<DualQuaternion> &node_se3,
	const mat34& world2camera,
	cudaTextureObject_t filter_foreground_mask,
	cudaStream_t stream
) {
	//Check the size
	SURFELWARP_CHECK(m_node_se3.Size() == node_se3.Size());
	
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_width, blk.x), divUp(m_image_height, blk.y));
	device::computeAlignmentErrorMapKernel<<<grid, blk, 0, stream>>>(
		m_depth_observation.vertex_map,
		m_depth_observation.normal_map,
		filter_foreground_mask,
		m_geometry_maps.reference_vertex_map,
		m_geometry_maps.reference_normal_map,
		m_geometry_maps.index_map,
		m_knn_map,
		node_se3.RawPtr(),
		m_project_intrinsic, world2camera,
		m_alignment_error_map.surface
	);
	
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

//First compute the error on node, then distribute them on map
void surfelwarp::DenseDepthHandler::ComputeNodewiseError(
	const DeviceArrayView<DualQuaternion> &node_se3, 
	const mat34 &world2camera,
	cudaTextureObject_t filter_foreground_mask, 
	cudaStream_t stream
) {
	const auto num_nodes = m_node_se3.Size();
	SURFELWARP_CHECK(node_se3.Size() == num_nodes);
	
	//Correct the size
	m_node_accumlate_error.ResizeArrayOrException(num_nodes);
	m_node_accumlate_weight.ResizeArrayOrException(num_nodes);
	
	//Clear the value
	cudaSafeCall(cudaMemsetAsync(m_node_accumlate_error.Ptr(), 0, sizeof(float) * num_nodes, stream));
	cudaSafeCall(cudaMemsetAsync(m_node_accumlate_weight.Ptr(), 0, sizeof(float) * num_nodes, stream));
	
	//First scatter the value to nodes
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_width, blk.x), divUp(m_image_height, blk.y));
	device::computeNodeAlignmentErrorFromMapKernel<<<grid, blk, 0, stream>>>(
		m_depth_observation.vertex_map,
		m_depth_observation.normal_map,
		filter_foreground_mask,
		m_geometry_maps.reference_vertex_map,
		m_geometry_maps.reference_normal_map,
		m_geometry_maps.index_map,
		m_knn_map,
		node_se3.RawPtr(),
		m_project_intrinsic, world2camera,
		m_node_accumlate_error.Ptr(),
		m_node_accumlate_weight.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::DenseDepthHandler::distributeNodeErrorOnMap(cudaStream_t stream) {
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_width, blk.x), divUp(m_image_height, blk.y));
	device::collectAlignmentErrorMapFromNodeKernel<<<grid, blk, 0, stream>>>(
		m_node_accumlate_error.Ptr(),
		m_node_accumlate_weight.Ptr(),
		m_geometry_maps.index_map,
		m_knn_map,
		m_alignment_error_map.surface
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}



void surfelwarp::DenseDepthHandler::ComputeAlignmentErrorMapFromNode(
	const DeviceArrayView<DualQuaternion>& node_se3, 
	const mat34 & world2camera,
	cudaTextureObject_t filter_foreground_mask,
	cudaStream_t stream
) {
	ComputeNodewiseError(node_se3, world2camera, filter_foreground_mask, stream);
	distributeNodeErrorOnMap(stream);
}