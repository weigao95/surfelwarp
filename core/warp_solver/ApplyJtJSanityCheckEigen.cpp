//
// Created by wei on 4/11/18.
//

#include "common/logging.h"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/ApplyJtJMatrixFreeHandler.h"

//Use for eigen
#include <Eigen/Eigen>
#include <Eigen/Sparse>


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

void surfelwarp::ApplyJtJHandlerMatrixFree::appendScalarCostJacobianTriplet(
	ScalarCostTerm2Jacobian& term2jacobian,
	unsigned row_offset,
	std::vector<Eigen::Triplet<float>>& jacobian_triplet,
	float term_weight
){
	//Download the required data
	std::vector<float4> knn_weight_array;
	std::vector<ushort4> knn_array;
	std::vector<TwistGradientOfScalarCost> twist_gradient_array;
	term2jacobian.knn_weight_array.Download(knn_weight_array);
	term2jacobian.knn_array.Download(knn_array);
	term2jacobian.twist_gradient_array.Download(twist_gradient_array);
	
	//Iterate over terms
	for(auto i = 0; i < knn_array.size(); i++) {
		const auto row_idx = i + row_offset;
		const ushort4 term_knn = knn_array[i];
		const float4 term_knn_weight = knn_weight_array[i];
		const TwistGradientOfScalarCost term_twist_gradient = twist_gradient_array[i];
		
		//Make them into flatten ptr
		const unsigned short* term_knn_arr = (const unsigned short*)(&term_knn);
		const float* term_knn_weight_arr = (const float*)(&term_knn_weight);
		const float* term_jacobian = (const float*)(&term_twist_gradient);
		
		//Iterates over knn
		for(auto j = 0; j < 4; j++) {
			auto node_idx = term_knn_arr[j];
			auto weight = term_knn_weight_arr[j];
			for(auto k = 0; k < 6; k++) {
				auto col_idx = k + 6 * node_idx;
				auto jacobian_value = term_weight * term_jacobian[k] * weight;
				jacobian_triplet.push_back(Eigen::Triplet<float>(row_idx, col_idx, jacobian_value));
			}
		}
	}
}


void surfelwarp::ApplyJtJHandlerMatrixFree::appendSmoothJacobianTriplet(std::vector<Eigen::Triplet<float>>& jacobian_triplet){
	//The row offset should be the number of depth terms
	auto num_depth_terms =  m_node2term_map.term_offset.DenseImageTermSize();
	auto row_offset = num_depth_terms;
	
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
		device::computeSmoothTermJacobian(xj, Ti, Tj, twist_gradient_i, twist_gradient_j);
		
		//Iterate over channels
		for(auto channel = 0; channel < 3; channel++) {
			auto row_idx = row_offset + k * 3 + channel;
			
			//The first node in this channel
			float* jacobian = (float*)(&twist_gradient_i[channel]);
			for(auto j = 0; j < 6; j++) {
				auto node_idx = node_ij.x;
				auto col_idx = 6 * node_idx + j;
				auto value = lambda_smooth * jacobian[j];
				jacobian_triplet.push_back(Eigen::Triplet<float>(row_idx, col_idx, value));
			}
			
			//The second node in this channel
			jacobian = (float*)(&twist_gradient_j[channel]);
			for(auto j = 0; j < 6; j++) {
				auto node_idx = node_ij.y;
				auto col_idx = 6 * node_idx + j;
				auto value = lambda_smooth * jacobian[j];
				jacobian_triplet.push_back(Eigen::Triplet<float>(row_idx, col_idx, value));
			}
		} // iteration over channel
	} // iteration over terms
}


void surfelwarp::ApplyJtJHandlerMatrixFree::appendFeatureJacobianTriplet(
	unsigned row_offset,
	std::vector<Eigen::Triplet<float>>& jacobian_triplet
){
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
		
		for(auto channel = 0; channel < 3; channel ++) {
			
			auto row_idx = row_offset + 3 * i + channel;
			const float* jacobian = (const float*)(&twist_gradient[channel]);
			
			for(auto j = 0; j < 4; j++) {
				auto node_idx = knn_ushort[j];
				auto node_weight = weight_float[j];
				for(auto k = 0; k < 6; k++) {
					auto col_idx = node_idx * 6 + k;
					auto value = lambda_feature * node_weight * jacobian[k];
					jacobian_triplet.push_back(Eigen::Triplet<float>(row_idx, col_idx, value));
				}
			}
		} // iterate over channels
	} // iterate over terms
}


void surfelwarp::ApplyJtJHandlerMatrixFree::applyJtJEigen(
	const std::vector<float> &x, unsigned num_scalar_terms,
	const std::vector<Eigen::Triplet<float>>& jacobian_triplet,
	std::vector<float> &jtj_x
){
	//Sanity check
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	SURFELWARP_CHECK_EQ(num_nodes * 6, x.size());
	SURFELWARP_CHECK_EQ(num_nodes * 6, jtj_x.size());
	
	//Construct the matrix
	Eigen::SparseMatrix<float> jacobian(num_scalar_terms, 6 * num_nodes);
	jacobian.setFromTriplets(jacobian_triplet.begin(), jacobian_triplet.end());
	
	//Construct the vector
	Eigen::VectorXf x_eigen; x_eigen.resize(x.size());
	for(auto i = 0; i < x.size(); i++) {
		x_eigen(i) = x[i];
	}
	
	//Perform jtj x
	Eigen::SparseMatrix<float> JtJ = Eigen::SparseMatrix<float>(jacobian.transpose()) * jacobian;
	Eigen::VectorXf jtj_x_eigen = JtJ * x_eigen;
	
	//Transfer back to std
	for(auto i = 0; i < jtj_x.size(); i++) {
		jtj_x[i] = jtj_x_eigen(i);
	}
}
