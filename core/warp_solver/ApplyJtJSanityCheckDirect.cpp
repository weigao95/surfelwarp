//
// Created by wei on 4/11/18.
//

#include "common/logging.h"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/ApplyJtJMatrixFreeHandler.h"


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

void surfelwarp::ApplyJtJHandlerMatrixFree::updateScalarJtJDotXDirect(
	const std::vector<float> &x,
	surfelwarp::ScalarCostTerm2Jacobian &term2jacobian,
	std::vector<float> &jtj_x,
	float term_weight_square
) {
	//Download the required data
	std::vector<float4> knn_weight_array;
	std::vector<ushort4> knn_array;
	std::vector<TwistGradientOfScalarCost> twist_gradient_array;
	term2jacobian.knn_weight_array.Download(knn_weight_array);
	term2jacobian.knn_array.Download(knn_array);
	term2jacobian.twist_gradient_array.Download(twist_gradient_array);
	
	//Iterate over terms
	for(auto i = 0; i < knn_array.size(); i++) {
		const ushort4 term_knn = knn_array[i];
		const float4 term_knn_weight = knn_weight_array[i];
		const TwistGradientOfScalarCost term_twist_gradient = twist_gradient_array[i];
		
		//Make them into flatten ptr
		const unsigned short* term_knn_arr = (const unsigned short*)(&term_knn);
		const float* term_knn_weight_arr = (const float*)(&term_knn_weight);
		
		for(auto blk_x = 0; blk_x < 4; blk_x++) {
			auto node_idx = term_knn_arr[blk_x];
			auto blk_x_weight = term_knn_weight_arr[blk_x];
			float accumlate_dot = 0.0f;

			for(auto blk_y = 0; blk_y < 4; blk_y++)
			{
				auto node_load_from = term_knn_arr[blk_y];
				auto blk_y_weight = term_knn_weight_arr[blk_y];
				const float* x_blk_y = &(x[node_load_from * 6]);
				accumlate_dot += blk_y_weight * term_twist_gradient.dot(x_blk_y);
			}

			//Dot with the weight of this value
			accumlate_dot *= blk_x_weight;

			//Write to output
			float* write_pos = &(jtj_x[6 * node_idx]);
			const float* jacobian = (const float*)(&term_twist_gradient);
			for(auto k = 0; k < 6; k++) {
				write_pos[k] += term_weight_square * accumlate_dot * jacobian[k];
			}
		} // iteration over the block
	} // iterate over terms
}

void surfelwarp::ApplyJtJHandlerMatrixFree::updateSmoothJtJDotXDirect(const std::vector<float> &x, std::vector<float> &jtj_x) {
	//Download the required data
	std::vector<ushort2> node_graph;
	std::vector<float4> node_coords;
	std::vector<DualQuaternion> node_se3;
	NodeGraphSmoothTerm2Jacobian smooth_term2jacobian = m_term2jacobian_map.smooth_term;
	smooth_term2jacobian.node_graph.Download(node_graph);
	smooth_term2jacobian.reference_node_coords.Download(node_coords);
	smooth_term2jacobian.node_se3.Download(node_se3);
	

	for(auto i = 0; i < node_graph.size(); i++) {
		ushort2 node_ij = node_graph[i];
		float4 xi = node_coords[node_ij.x];
		float4 xj = node_coords[node_ij.y];
		mat34 Ti = node_se3[node_ij.x].se3_matrix();
		mat34 Tj = node_se3[node_ij.y].se3_matrix();
		TwistGradientOfScalarCost twist_gradient_i[3];
		TwistGradientOfScalarCost twist_gradient_j[3];
		float residual[3];
		device::computeSmoothTermJacobianResidual(xi, xj, Ti, Tj, residual, twist_gradient_i, twist_gradient_j);
		
		//Make it flatten
		unsigned short* node_ij_arr = (unsigned short*)(&node_ij);
		
		for(auto channel = 0; channel < 3; channel++) {
			//For the first node
			float accumlate_dot = 0.0f;
			for(auto k = 0; k < 2; k++) {
				auto node_load_from = node_ij_arr[k];
				const float* x_blk_k = &(x[node_load_from * 6]);
				if(k == 0) accumlate_dot += twist_gradient_i[channel].dot(x_blk_k);
				else accumlate_dot += twist_gradient_j[channel].dot(x_blk_k);
			}
			
			//Fill for the first node
			auto node_fill_idx = node_ij_arr[0];
			float* write_pos = &(jtj_x[6 * node_fill_idx]);
			const float* jacobian = (const float*)(&twist_gradient_i[channel]);
			for(auto k = 0; k < 6; k++) {
				write_pos[k] += lambda_smooth_square * accumlate_dot * jacobian[k];
			}
			
			//Fill for the second node
			node_fill_idx = node_ij_arr[1];
			write_pos = &(jtj_x[6 * node_fill_idx]);
			jacobian = (const float*)(&twist_gradient_j[channel]);
			for(auto k = 0; k < 6; k++) {
				write_pos[k] += lambda_smooth_square * accumlate_dot * jacobian[k];
			}
			
		} // iterate over all 3 channels
	} // iterate over all terms
}



void surfelwarp::ApplyJtJHandlerMatrixFree::updateFeatureJtJDotXDirect(
	const std::vector<float> &x,
	std::vector<float> &jtj_x
) {
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
	
	//Iterates over terms
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
		
		//Iterate over channels
		for(auto channel = 0; channel < 3; channel++) {
			//Perform dot product
			float accumlate_dot = 0.0f;
			for(auto blk_y = 0; blk_y < 4; blk_y++)
			{
				auto node_load_from = knn_ushort[blk_y];
				auto blk_y_weight = weight_float[blk_y];
				const float* x_blk_y = &(x[node_load_from * 6]);
				accumlate_dot += blk_y_weight * twist_gradient[channel].dot(x_blk_y);
			}
			
			//Fill the value to correct position
			const float* jacobian = (const float*)(&twist_gradient[channel]);
			for(auto k = 0; k < 4; k++) {
				auto node_write = knn_ushort[k];
				auto node_write_weight = weight_float[k];
				float* write_pos = &(jtj_x[6 * node_write]);
				for(auto j = 0; j < 6; j++) {
					write_pos[j] += lambda_feature_square * node_write_weight * accumlate_dot * jacobian[j];
				}
			}
		} // iterate over channels
	} // iterate over terms
}