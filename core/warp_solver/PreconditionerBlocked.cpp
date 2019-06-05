//
// Created by wei on 4/7/18.
//
#include "common/logging.h"
#include "common/sanity_check.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/PreconditionerRhsBuilder.h"
#include "core/warp_solver/solver_constants.h"

void surfelwarp::PreconditionerRhsBuilder::updateScalarCostJtJDiagonalHost(
	std::vector<float> &jtj_flatten,
	ScalarCostTerm2Jacobian term2jacobian,
	float term_weight_square
) {
	const auto num_nodes = (m_node2term_map.offset.Size() - 1);
	SURFELWARP_CHECK_EQ(jtj_flatten.size(), num_nodes * 36);
	
	//Download the depth jacobian
	std::vector<ushort4> knn_array;
	std::vector<float4> knn_weight_array;
	std::vector<TwistGradientOfScalarCost> twist_gradient_array;
	const auto jacobian_collect = term2jacobian;
	jacobian_collect.knn_array.Download(knn_array);
	jacobian_collect.knn_weight_array.Download(knn_weight_array);
	jacobian_collect.twist_gradient_array.Download(twist_gradient_array);
	
	//Simple sanity check
	SURFELWARP_CHECK_EQ(knn_array.size(), knn_weight_array.size());
	SURFELWARP_CHECK_EQ(knn_array.size(), twist_gradient_array.size());
	
	//Iterate through costs
	for(auto i = 0; i < knn_array.size(); i++) {
		const unsigned short* knn = (unsigned short*)(&knn_array[i]);
		const float* knn_weight = (const float*)(&knn_weight_array[i]);
		const float* jacobian = (float*)&(twist_gradient_array[i]);
		for(auto j = 0; j < 4; j++) {
			unsigned short node_idx = knn[j];
			float weight = knn_weight[j];
			float* jtj = &jtj_flatten[node_idx * 36];
			for (int jac_row = 0; jac_row < 6; jac_row++) //Do computation of jacobian
			{
				jtj[6 * jac_row + 0] += term_weight_square * weight * weight * jacobian[0] * jacobian[jac_row];
				jtj[6 * jac_row + 1] += term_weight_square * weight * weight * jacobian[1] * jacobian[jac_row];
				jtj[6 * jac_row + 2] += term_weight_square * weight * weight * jacobian[2] * jacobian[jac_row];
				jtj[6 * jac_row + 3] += term_weight_square * weight * weight * jacobian[3] * jacobian[jac_row];
				jtj[6 * jac_row + 4] += term_weight_square * weight * weight * jacobian[4] * jacobian[jac_row];
				jtj[6 * jac_row + 5] += term_weight_square * weight * weight * jacobian[5] * jacobian[jac_row];
			}
		}
	}
}


void surfelwarp::PreconditionerRhsBuilder::updateSmoothJtJDiagonalHost(std::vector<float> &jtj_flatten) {
	//Download the required data
	std::vector<ushort2> node_graph;
	std::vector<float4> node_coords;
	std::vector<DualQuaternion> node_se3;
	NodeGraphSmoothTerm2Jacobian smooth_term2jacobian = m_term2jacobian_map.smooth_term;
	smooth_term2jacobian.node_graph.Download(node_graph);
	smooth_term2jacobian.reference_node_coords.Download(node_coords);
	smooth_term2jacobian.node_se3.Download(node_se3);
	
	//Iterates through all node pairs
	for(auto i = 0; i < node_graph.size(); i++) {
		ushort2 node_ij = node_graph[i];
		float4 xj = node_coords[node_ij.y];
		mat34 Ti = node_se3[node_ij.x].se3_matrix();
		mat34 Tj = node_se3[node_ij.y].se3_matrix();
		TwistGradientOfScalarCost twist_gradient_i[3];
		TwistGradientOfScalarCost twist_gradient_j[3];
		device::computeSmoothTermJacobian(xj, Ti, Tj, twist_gradient_i, twist_gradient_j);
		
		//First fill the node i
		float* jtj = &jtj_flatten[node_ij.x * 36];
		for(auto channel = 0; channel < 3; channel++) {
			float* jacobian = (float*)(&twist_gradient_i[channel]);
			for (int jac_row = 0; jac_row < 6; jac_row++) {
				jtj[6 * jac_row + 0] += lambda_smooth_square * jacobian[0] * jacobian[jac_row];
				jtj[6 * jac_row + 1] += lambda_smooth_square * jacobian[1] * jacobian[jac_row];
				jtj[6 * jac_row + 2] += lambda_smooth_square * jacobian[2] * jacobian[jac_row];
				jtj[6 * jac_row + 3] += lambda_smooth_square * jacobian[3] * jacobian[jac_row];
				jtj[6 * jac_row + 4] += lambda_smooth_square * jacobian[4] * jacobian[jac_row];
				jtj[6 * jac_row + 5] += lambda_smooth_square * jacobian[5] * jacobian[jac_row];
			}
		}
		
		//Then fill node j
		jtj = &jtj_flatten[node_ij.y * 36];
		for(auto channel = 0; channel < 3; channel++) {
			float* jacobian = (float*)(&twist_gradient_j[channel]);
			for (int jac_row = 0; jac_row < 6; jac_row++) {
				jtj[6 * jac_row + 0] += lambda_smooth_square * jacobian[0] * jacobian[jac_row];
				jtj[6 * jac_row + 1] += lambda_smooth_square * jacobian[1] * jacobian[jac_row];
				jtj[6 * jac_row + 2] += lambda_smooth_square * jacobian[2] * jacobian[jac_row];
				jtj[6 * jac_row + 3] += lambda_smooth_square * jacobian[3] * jacobian[jac_row];
				jtj[6 * jac_row + 4] += lambda_smooth_square * jacobian[4] * jacobian[jac_row];
				jtj[6 * jac_row + 5] += lambda_smooth_square * jacobian[5] * jacobian[jac_row];
			}
		}
	}
}


//This verion is deprecated, thus move here for sanity check purpose only
namespace surfelwarp { namespace device {
	
	__host__ __forceinline__ void computePointToPointICPTermJacobian(
		const float4& depth_vertex_confid,
		const float4& can_vertex4,
		const ushort4& knn, const float4& knn_weight,
		//The warp field
		const DualQuaternion* device_warp_field,
		const mat34& camera2world,
		//The output
		TwistGradientOfScalarCost* twist_gradient //[3]
	) {
		//Correct the size
		const float3 can_vertex = make_float3(can_vertex4.x, can_vertex4.y, can_vertex4.z);
		const float3 depth_vertex = make_float3(depth_vertex_confid.x, depth_vertex_confid.y, depth_vertex_confid.z);

		//Warp it
		DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
		const mat34 se3 = dq_average.se3_matrix();
		const float3 warped_vertex = se3.rot * can_vertex + se3.trans;

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

void surfelwarp::PreconditionerRhsBuilder::updateFeatureJtJDiagonalHost(std::vector<float> &jtj_flatten) {
	//Download the required data
	auto term2jacobian_dev = m_term2jacobian_map.sparse_feature_term;
	std::vector<float4> depth_vertex_arr, reference_vertex_arr, knn_weight_arr;
	std::vector<ushort4> knn_arr;
	std::vector<DualQuaternion> node_se3_arr;
	term2jacobian_dev.target_vertex.Download(depth_vertex_arr);
	term2jacobian_dev.reference_vertex.Download(reference_vertex_arr);
	term2jacobian_dev.knn.Download(knn_arr);
	term2jacobian_dev.knn_weight.Download(knn_weight_arr);
	term2jacobian_dev.node_se3.Download(node_se3_arr);
	
	//Simple check
	SURFELWARP_CHECK_EQ(knn_arr.size(), depth_vertex_arr.size());
	SURFELWARP_CHECK_EQ(knn_arr.size(), reference_vertex_arr.size());
	SURFELWARP_CHECK_EQ(knn_arr.size(), knn_weight_arr.size());
	
	//Iterates through terms
	for(auto i = 0; i < knn_arr.size(); i++) {
		float4 depth_vertex = depth_vertex_arr[i];
		float4 ref_vertex = reference_vertex_arr[i];
		ushort4 knn = knn_arr[i];
		float4 knn_weight = knn_weight_arr[i];
		TwistGradientOfScalarCost twist_gradient[3];
		device::computePointToPointICPTermJacobian(
			depth_vertex,
			ref_vertex,
			knn, knn_weight,
			node_se3_arr.data(),
			mat34::identity(),
			twist_gradient
		);
		
		//Fill each knn
		unsigned short* knn_ushort = (unsigned short*)(&knn);
		float* weight_float = (float*)(&knn_weight);
		for(auto j = 0; j < 4; j++) {
			unsigned short node_idx = knn_ushort[j];
			float weight = weight_float[j];
			//First fill the node i
			float* jtj = &jtj_flatten[node_idx * 36];
			for(auto channel = 0; channel < 3; channel++) {
				float* jacobian = (float*)(&twist_gradient[channel]);
				for (int jac_row = 0; jac_row < 6; jac_row++) {
					jtj[6 * jac_row + 0] += lambda_feature_square * weight * weight * jacobian[0] * jacobian[jac_row];
					jtj[6 * jac_row + 1] += lambda_feature_square * weight * weight * jacobian[1] * jacobian[jac_row];
					jtj[6 * jac_row + 2] += lambda_feature_square * weight * weight * jacobian[2] * jacobian[jac_row];
					jtj[6 * jac_row + 3] += lambda_feature_square * weight * weight * jacobian[3] * jacobian[jac_row];
					jtj[6 * jac_row + 4] += lambda_feature_square * weight * weight * jacobian[4] * jacobian[jac_row];
					jtj[6 * jac_row + 5] += lambda_feature_square * weight * weight * jacobian[5] * jacobian[jac_row];
				}
			}
		} // End of 4 knn iteration
	} // End of term iteration
}

void surfelwarp::PreconditionerRhsBuilder::diagonalPreconditionerSanityCheck() {
	LOG(INFO) << "Check the diagonal elements of JtJ";
	
	//Download the device value
	std::vector<float> diagonal_blks_dev;
	diagonal_blks_dev.resize(m_block_preconditioner.ArraySize());
	m_block_preconditioner.ArrayView().Download(diagonal_blks_dev);
	SURFELWARP_CHECK_EQ(diagonal_blks_dev.size(), 36 * (m_node2term_map.offset.Size() - 1));
	
	//Download the node2term map
	std::vector<unsigned> node_offset;
	std::vector<unsigned> term_index_value;
	m_node2term_map.offset.Download(node_offset);
	m_node2term_map.term_index.Download(term_index_value);
	
	//Check the dense depth terms
	std::vector<float> jtj_diagonal;
	jtj_diagonal.resize(diagonal_blks_dev.size());
	for(auto i = 0; i < jtj_diagonal.size(); i++) {
		jtj_diagonal[i] = 0.0f;
	}
	
	//Compute the depth term
	updateScalarCostJtJDiagonalHost(jtj_diagonal, m_term2jacobian_map.dense_depth_term);
	updateScalarCostJtJDiagonalHost(jtj_diagonal, m_term2jacobian_map.density_map_term, lambda_density_square);
	updateScalarCostJtJDiagonalHost(jtj_diagonal, m_term2jacobian_map.foreground_mask_term, lambda_foreground_square);
	updateSmoothJtJDiagonalHost(jtj_diagonal);
	updateFeatureJtJDiagonalHost(jtj_diagonal);
	
	//Check it
	auto relative_err = maxRelativeError(diagonal_blks_dev, jtj_diagonal, 0.01f);
	
	LOG(INFO) << "The relative error is " << relative_err;
}
