#include "common/ConfigParser.h"
#include "common/sanity_check.h"
#include "common/logging.h"
#include "core/warp_solver/DenseDepthHandler.h"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	__global__ void computeDenseDepthJacobianKernel(
		cudaTextureObject_t	depth_vertex_confid_map,
		cudaTextureObject_t depth_normal_radius_map,
		cudaTextureObject_t reference_vertex_map,
		const DeviceArrayView<ushort4> correspond_pixel_pair,
		const ushort4* pixel_knn,
		const float4* pixel_knn_weight,
		const DualQuaternion* node_se3,
		const mat34 camera2world,
		//Output
		TwistGradientOfScalarCost* twist_gradient,
		float* residual
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < correspond_pixel_pair.Size())
		{
			//Request informations
			const ushort4 pixel_pair = correspond_pixel_pair[idx];
			const ushort4 knn = pixel_knn[idx];
			const float4 knn_weight = pixel_knn_weight[idx];
			const float4 reference_vertex = tex2D<float4>(reference_vertex_map, pixel_pair.x, pixel_pair.y);
			const float4 depth_vertex_confid = tex2D<float4>(depth_vertex_confid_map, pixel_pair.z, pixel_pair.w);
			const float4 depth_normal_radius = tex2D<float4>(depth_normal_radius_map, pixel_pair.z, pixel_pair.w);

			//Compute it
			computePointToPlaneICPTermJacobianResidual(
				depth_vertex_confid,
				depth_normal_radius, 
				reference_vertex, 
				knn, knn_weight,
				//Deformation
				node_se3, 
				camera2world,
				//The output
				twist_gradient[idx],
				residual[idx]
			);
		}
	}


	//This version needs to first check whether the given vertex will result in a match. If there is
	//a correspondence, the fill in the jacobian and residual values, else mark the value to zero. 
	//The input is only depended on the SE(3) of the nodes, which can be updated without rebuild the index
	__global__ void computeDenseDepthJacobianKernel(
		cudaTextureObject_t depth_vertex_confid_map,
		cudaTextureObject_t depth_normal_radius_map,
		cudaTextureObject_t reference_vertex_map,
		cudaTextureObject_t reference_normal_map,
		unsigned img_rows, unsigned img_cols,
		//The potential matched pixels and their knn
		DeviceArrayView<ushort2> potential_matched_pixels,
		const ushort4* potential_matched_knn,
		const float4* potential_matched_knn_weight,
		//The deformation
		const DualQuaternion* node_se3,
		const mat34 world2camera,
		const Intrinsic intrinsic,
		//The output
		TwistGradientOfScalarCost* twist_gradient,
		float* residual
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx >= potential_matched_pixels.Size()) return;

		//These value will definited be written to global memory
		float pixel_residual = 0.0f;
		float pixel_gradient[6] = {0};
		TwistGradientOfScalarCost* pixel_twist = (TwistGradientOfScalarCost*)pixel_gradient;

		//Now, query the pixel, knn and their weight
		const ushort2 potential_pixel = potential_matched_pixels[idx];
		const ushort4 knn = potential_matched_knn[idx];
		const float4 knn_weight = potential_matched_knn_weight[idx];

		//Get the vertex
		const float4 can_vertex4 = tex2D<float4>(reference_vertex_map, potential_pixel.x, potential_pixel.y);
		const float4 can_normal4 = tex2D<float4>(reference_normal_map, potential_pixel.x, potential_pixel.y);
		DualQuaternion dq_average = averageDualQuaternion(node_se3, knn, knn_weight);
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

		if (img_coord.x >= 0 && img_coord.x < img_cols && img_coord.y >= 0 && img_coord.y < img_rows)
		{
			//Query the depth image
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


			//This pair is valid, compute the jacobian and residual
			if(valid_pair) {
				pixel_residual = dot(depth_normal, warped_vertex_camera - depth_vertex);
				pixel_twist->translation = world2camera.rot.transpose_dot(depth_normal); // depth_world_normal
				pixel_twist->rotation = cross(warped_vertex, pixel_twist->translation); // cross(warp_vertex, depth_world_normal)
			}
		} // This pixel is projected inside

		//Write it to global memory
		residual[idx] = pixel_residual;
		twist_gradient[idx] = *pixel_twist;
	}

} // namespace device
} // namespace surfelwarp



/* The method and buffer for gradient computation
 */
void surfelwarp::DenseDepthHandler::ComputeJacobianTermsFreeIndex(cudaStream_t stream) {
	//Correct the size of array
	m_term_twist_gradient.ResizeArrayOrException(m_valid_pixel_pairs.ArraySize());
	m_term_residual.ResizeArrayOrException(m_valid_pixel_pairs.ArraySize());

	//Invoke kernel
	dim3 blk(128);
	dim3 grid(divUp(m_valid_pixel_pairs.ArraySize(), blk.x));
	device::computeDenseDepthJacobianKernel<<<grid, blk, 0, stream>>>(
		m_depth_observation.vertex_map,
		m_depth_observation.normal_map, 
		m_geometry_maps.reference_vertex_map,
		//Pixels and KNN
		m_valid_pixel_pairs.ArrayView(),
		m_dense_depth_knn.Ptr(),
		m_dense_depth_knn_weight.Ptr(),
		//The deformation
		m_node_se3.RawPtr(), 
		m_camera2world,
		//Output
		m_term_twist_gradient.Ptr(),
		m_term_residual.Ptr()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::DenseDepthHandler::ComputeJacobianTermsFixedIndex(cudaStream_t stream) {
	m_term_residual.ResizeArrayOrException(m_potential_pixels_knn.pixels.Size());
	m_term_twist_gradient.ResizeArrayOrException(m_potential_pixels_knn.pixels.Size());
	
	dim3 blk(128);
	dim3 grid(divUp(m_potential_pixels_knn.pixels.Size(), blk.x));
	device::computeDenseDepthJacobianKernel<<<grid, blk, 0, stream>>>(
		m_depth_observation.vertex_map,
		m_depth_observation.normal_map,
		m_geometry_maps.reference_vertex_map,
		m_geometry_maps.reference_normal_map,
		m_knn_map.Rows(), m_knn_map.Cols(),
		//The pixel pairs and knn
		m_potential_pixels_knn.pixels,
		m_potential_pixels_knn.node_knn.RawPtr(),
		m_potential_pixels_knn.knn_weight.RawPtr(),
		//The deformation
		m_node_se3.RawPtr(),
		m_world2camera,
		m_project_intrinsic,
		//The output
		m_term_twist_gradient.Ptr(),
		m_term_residual.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


surfelwarp::DenseDepthTerm2Jacobian surfelwarp::DenseDepthHandler::Term2JacobianMap() const {
	DenseDepthTerm2Jacobian term2jacobian;
	term2jacobian.knn_array = m_potential_pixels_knn.node_knn;
	term2jacobian.knn_weight_array = m_potential_pixels_knn.knn_weight;
	term2jacobian.residual_array = m_term_residual.ArrayView();
	term2jacobian.twist_gradient_array = m_term_twist_gradient.ArrayView();
	term2jacobian.check_size();
	
	//Check correct
	return term2jacobian;
}