//
// Created by wei on 4/9/18.
//

#include "common/logging.h"
#include "common/sanity_check.h"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/PreconditionerRhsBuilder.h"

/* The interface function for JtResidual
 */
void surfelwarp::PreconditionerRhsBuilder::ComputeJtResidual(cudaStream_t stream) {
	ComputeJtResidualIndexed(stream);
	//ComputeJtResidualAtomic(stream);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Sanity check
	//jacobianTransposeResidualSanityCheck();
}



/* The method for sanity check
 */
void surfelwarp::PreconditionerRhsBuilder::updateScalarCostJtResidualHost(
	std::vector<float> &jt_residual,
	ScalarCostTerm2Jacobian term2jacobian,
	float term_weight_square
) {
	//Check the size
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	SURFELWARP_CHECK_EQ(num_nodes * 6, jt_residual.size());
	
	//Download the data
	std::vector<ushort4> knn_array;
	std::vector<float4> knn_weight_array;
	std::vector<float> residual_array;
	std::vector<TwistGradientOfScalarCost> twist_gradient_array;
	term2jacobian.knn_array.Download(knn_array);
	term2jacobian.knn_weight_array.Download(knn_weight_array);
	term2jacobian.twist_gradient_array.Download(twist_gradient_array);
	term2jacobian.residual_array.Download(residual_array);
	
	//Iterates over terms
	for(auto i = 0; i < knn_array.size(); i++) {
		const unsigned short* knn = (unsigned short*)(&knn_array[i]);
		const float* knn_weight = (const float*)(&knn_weight_array[i]);
		const float* jacobian = (float*)&(twist_gradient_array[i]);
		const float residual = residual_array[i];
		for(auto j = 0; j < 4; j++) {
			unsigned short node_idx = knn[j];
			float weight = knn_weight[j];
			float* jt_r_node = &jt_residual[node_idx * 6];
			for(auto k = 0; k < 6; k++) {
				jt_r_node[k] += - term_weight_square * weight * residual * jacobian[k];
			}
		}
	}
}


void surfelwarp::PreconditionerRhsBuilder::updateSmoothJtResidualHost(std::vector<float>& jt_residual)
{
	//Download the required data
	std::vector<ushort2> node_graph;
	std::vector<float4> node_coords;
	std::vector<DualQuaternion> node_se3;
	NodeGraphSmoothTerm2Jacobian smooth_term2jacobian = m_term2jacobian_map.smooth_term;
	smooth_term2jacobian.node_graph.Download(node_graph);
	smooth_term2jacobian.reference_node_coords.Download(node_coords);
	smooth_term2jacobian.node_se3.Download(node_se3);

	//Iterates through all node pairs
	for(auto k = 0; k < node_graph.size(); k++) {
		ushort2 node_ij = node_graph[k];
		float4 xi = node_coords[node_ij.x];
		float4 xj = node_coords[node_ij.y];
		mat34 Ti = node_se3[node_ij.x].se3_matrix();
		mat34 Tj = node_se3[node_ij.y].se3_matrix();
		TwistGradientOfScalarCost twist_gradient_i[3];
		TwistGradientOfScalarCost twist_gradient_j[3];
		float residual[3];
		device::computeSmoothTermJacobianResidual(xi, xj, Ti, Tj, residual, twist_gradient_i, twist_gradient_j);
		
		//First fill the node i
		float* jt_r_node = &jt_residual[node_ij.x * 6];
		for(auto channel = 0; channel < 3; channel++) {
			float* jacobian = (float*)(&twist_gradient_i[channel]);
			for(auto j = 0; j < 6; j++) {
				jt_r_node[j] += - lambda_smooth_square * residual[channel] * jacobian[j];
			}
		}
		
		//Then fill node j
		jt_r_node = &jt_residual[node_ij.y * 6];
		for(auto channel = 0; channel < 3; channel++) {
			float* jacobian = (float*)(&twist_gradient_j[channel]);
			for(auto j = 0; j < 6; j++) {
				jt_r_node[j] += - lambda_smooth_square * residual[channel] * jacobian[j];
			}
		} 
	}
}


//This method is actually deprecated, thus move it to here for immediate use
namespace surfelwarp { namespace device {
	
	__host__ __forceinline__ void computePointToPointICPTermJacobianResidual(
		const float4& depth_vertex_confid,
		const float4& can_vertex4,
		const ushort4& knn, const float4& knn_weight,
		//The warp field
		const DualQuaternion* device_warp_field,
		const mat34& camera2world,
		//The output
		float* residual, //[3]
		TwistGradientOfScalarCost* twist_gradient //[3]
	) {
		//Correct the size
		const float3 can_vertex = make_float3(can_vertex4.x, can_vertex4.y, can_vertex4.z);
		const float3 depth_vertex = make_float3(depth_vertex_confid.x, depth_vertex_confid.y, depth_vertex_confid.z);

		//Warp it
		DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
		const mat34 se3 = dq_average.se3_matrix();
		const float3 warped_vertex = se3.rot * can_vertex + se3.trans;

		//Warp the depth vertex to world frame
		const float3 depth_world_vertex = camera2world.rot * depth_vertex + camera2world.trans;

		//Compute the residual
		residual[0] = warped_vertex.x - depth_world_vertex.x;
		residual[1] = warped_vertex.y - depth_world_vertex.y;
		residual[2] = warped_vertex.z - depth_world_vertex.z;

		//Compute the gradient
		twist_gradient[0].translation = make_float3(1.0f, 0.0f, 0.0f);
		twist_gradient[1].translation = make_float3(0.0f, 1.0f, 0.0f);
		twist_gradient[2].translation = make_float3(0.0f, 0.0f, 1.0f);
		twist_gradient[0].rotation = make_float3(0.0f, warped_vertex.z, -warped_vertex.y);
		twist_gradient[1].rotation = make_float3(-warped_vertex.z, 0.0f, warped_vertex.x);
		twist_gradient[2].rotation = make_float3(warped_vertex.y, -warped_vertex.x, 0.0f);
	}


} // device
} // surfelwarp


void surfelwarp::PreconditionerRhsBuilder::updateFeatureJtResidualHost(std::vector<float> &jt_residual) {
	const auto term2jacobian = m_term2jacobian_map.sparse_feature_term;
	std::vector<float4> ref_vertex_arr, depth_vertex_arr, knn_weight_arr;
	std::vector<ushort4> knn_arr;
	std::vector<DualQuaternion> node_se3;
	term2jacobian.node_se3.Download(node_se3);
	term2jacobian.reference_vertex.Download(ref_vertex_arr);
	term2jacobian.target_vertex.Download(depth_vertex_arr);
	term2jacobian.knn_weight.Download(knn_weight_arr);
	term2jacobian.knn.Download(knn_arr);
	
	for(auto i = 0; i < knn_arr.size(); i++) {
		float4 depth_vertex = depth_vertex_arr[i];
		float4 ref_vertex = ref_vertex_arr[i];
		ushort4 knn = knn_arr[i];
		float4 knn_weight = knn_weight_arr[i];
		float residual[3];
		TwistGradientOfScalarCost twist_gradient[3];
		device::computePointToPointICPTermJacobianResidual(
			depth_vertex,
			ref_vertex,
			knn, knn_weight,
			node_se3.data(),
			mat34::identity(),
			residual,
			twist_gradient
		);
		
		//For the knns of this node
		unsigned short* flatten_knn = (unsigned short*)(&knn);
		float* flatten_weight = (float*)(&knn_weight);
		for(auto j = 0; j < 4; j++) {
			auto node_idx = flatten_knn[j];
			auto weight = flatten_weight[j];
			float* jt_r_node = &jt_residual[node_idx * 6];
			for(auto channel = 0; channel < 3; channel++) {
				float* jacobian = (float*)(&twist_gradient[channel]);
				for(auto k = 0; k < 6; k++) {
					jt_r_node[k] += - lambda_feature_square * weight * residual[channel] * jacobian[k];
				}
			}
		}
	}
}

void surfelwarp::PreconditionerRhsBuilder::jacobianTransposeResidualSanityCheck() {
	LOG(INFO) << "Check the elements of Jt Residual";

	//Compute the value at host
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	std::vector<float> jt_residual;
	jt_residual.resize(num_nodes * 6);
	for(auto i = 0; i < jt_residual.size(); i++) {
		jt_residual[i] = 0.0f;
	}
	
	//Compute each terms
	updateScalarCostJtResidualHost(jt_residual, m_term2jacobian_map.dense_depth_term);
	updateScalarCostJtResidualHost(jt_residual, m_term2jacobian_map.density_map_term, lambda_density_square);
	updateScalarCostJtResidualHost(jt_residual, m_term2jacobian_map.foreground_mask_term, lambda_foreground_square);
	updateSmoothJtResidualHost(jt_residual);
	updateFeatureJtResidualHost(jt_residual);

	//Download the results from device
	std::vector<float> jt_residual_dev;
	m_jt_residual.ArrayView().Download(jt_residual_dev);
	SURFELWARP_CHECK_EQ(jt_residual.size(), jt_residual_dev.size());

	//Check it
	auto relative_err = maxRelativeError(jt_residual, jt_residual_dev, 0.001f, true);
	LOG(INFO) << "The relative error is " << relative_err;
}





