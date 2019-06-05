//
// Created by wei on 4/18/18.
//

#include "common/ConfigParser.h"
#include "common/Constants.h"
#include "common/sanity_check.h"
#include "pcg_solver/ApplySpMVBinBlockCSR.h"
#include "core/warp_solver/solver_encode.h"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/JtJMaterializer.h"
#include <map>


void surfelwarp::JtJMaterializer::updateScalarCostJtJBlockHost(
	std::vector<float> &jtj_flatten,
	ScalarCostTerm2Jacobian term2jacobian,
	const float term_weight_square
) {
	const auto num_nodepairs = m_nodepair2term_map.nodepair_term_range.Size();
	SURFELWARP_CHECK_EQ(num_nodepairs * 36, jtj_flatten.size());
	
	//Download the jacobian
	std::vector<ushort4> knn_array;
	std::vector<float4> knn_weight_array;
	std::vector<TwistGradientOfScalarCost> twist_gradient_array;
	const auto jacobian_collect = term2jacobian;
	jacobian_collect.knn_array.Download(knn_array);
	jacobian_collect.knn_weight_array.Download(knn_weight_array);
	jacobian_collect.twist_gradient_array.Download(twist_gradient_array);
	
	//Prepare the map from node pair to index
	std::map<unsigned, unsigned> pair2index;
	pair2index.clear();
	std::vector<unsigned> encoded_pair_vec;
	m_nodepair2term_map.encoded_nodepair.Download(encoded_pair_vec);
	for(auto i = 0; i < encoded_pair_vec.size(); i++) {
		auto encoded_pair = encoded_pair_vec[i];
		//There should not be duplicate
		auto iter = pair2index.find(encoded_pair);
		assert(iter == pair2index.end());
		
		//insert it
		pair2index.insert(std::make_pair(encoded_pair, i));
	}
	
	//Simple sanity check
	SURFELWARP_CHECK_EQ(knn_array.size(), knn_weight_array.size());
	SURFELWARP_CHECK_EQ(knn_array.size(), twist_gradient_array.size());
	
	//Iterate through costs
	for(auto term_idx = 0; term_idx < knn_array.size(); term_idx++) {
		const unsigned short* knn = (unsigned short*)(&knn_array[term_idx]);
		const float* knn_weight = (const float*)(&knn_weight_array[term_idx]);
		const float* jacobian = (float*)&(twist_gradient_array[term_idx]);
		for(auto i = 0; i < 4; i++) {
			unsigned short node_i = knn[i];
			float weight_i = knn_weight[i];
			for(auto j = 0; j < 4; j++) {
				if(i == j) continue;
				unsigned short node_j = knn[j];
				float weight_j = knn_weight[j];
				auto encoded_ij = encode_nodepair(node_i, node_j);
				auto iter = pair2index.find(encoded_ij);
				if(iter != pair2index.end()) {
					unsigned index = iter->second;
					float* jtj = &jtj_flatten[index * 36];
					for (int jac_row = 0; jac_row < 6; jac_row++) //Do computation of jacobian
					{
						jtj[6 * jac_row + 0] += term_weight_square * weight_i * weight_j * jacobian[0] * jacobian[jac_row];
						jtj[6 * jac_row + 1] += term_weight_square * weight_i * weight_j * jacobian[1] * jacobian[jac_row];
						jtj[6 * jac_row + 2] += term_weight_square * weight_i * weight_j * jacobian[2] * jacobian[jac_row];
						jtj[6 * jac_row + 3] += term_weight_square * weight_i * weight_j * jacobian[3] * jacobian[jac_row];
						jtj[6 * jac_row + 4] += term_weight_square * weight_i * weight_j * jacobian[4] * jacobian[jac_row];
						jtj[6 * jac_row + 5] += term_weight_square * weight_i * weight_j * jacobian[5] * jacobian[jac_row];
					}
				} else {
					LOG(FATAL) << "Cannot find the index for node " << node_i << " and " << node_j << " pair!";
				}
			}
			
		}
	}
}

void surfelwarp::JtJMaterializer::updateSmoothCostJtJBlockHost(std::vector<float> &jtj_flatten) {
	const auto num_nodepairs = m_nodepair2term_map.nodepair_term_range.Size();
	SURFELWARP_CHECK_EQ(num_nodepairs * 36, jtj_flatten.size());
	
	//Download the required data
	std::vector<ushort2> node_graph;
	std::vector<float4> node_coords;
	std::vector<DualQuaternion> node_se3;
	NodeGraphSmoothTerm2Jacobian smooth_term2jacobian = m_term2jacobian_map.smooth_term;
	smooth_term2jacobian.node_graph.Download(node_graph);
	smooth_term2jacobian.reference_node_coords.Download(node_coords);
	smooth_term2jacobian.node_se3.Download(node_se3);
	
	//Prepare the map from node pair to index
	std::map<unsigned, unsigned> pair2index;
	pair2index.clear();
	std::vector<unsigned> encoded_pair_vec;
	m_nodepair2term_map.encoded_nodepair.Download(encoded_pair_vec);
	for(auto i = 0; i < encoded_pair_vec.size(); i++) {
		auto encoded_pair = encoded_pair_vec[i];
		//There should not be duplicate
		auto iter = pair2index.find(encoded_pair);
		assert(iter == pair2index.end());
		
		//insert it
		pair2index.insert(std::make_pair(encoded_pair, i));
	}
	
	for(auto term_idx = 0; term_idx < node_graph.size(); term_idx++) {
		ushort2 node_ij = node_graph[term_idx];
		float4 xj = node_coords[node_ij.y];
		mat34 Ti = node_se3[node_ij.x].se3_matrix();
		mat34 Tj = node_se3[node_ij.y].se3_matrix();
		TwistGradientOfScalarCost twist_gradient_i[3];
		TwistGradientOfScalarCost twist_gradient_j[3];
		device::computeSmoothTermJacobian(xj, Ti, Tj, twist_gradient_i, twist_gradient_j);
		
		//First fill the (node i, node j)
		const unsigned node_i = node_ij.x;
		const unsigned node_j = node_ij.y;
		const float* encoded_jacobian_i = (const float*)(twist_gradient_i);
		const float* encoded_jacobian_j = (const float*)(twist_gradient_j);
		auto encoded_ij = encode_nodepair(node_i, node_j);
		auto iter = pair2index.find(encoded_ij);
		if(iter != pair2index.end()) {
			unsigned index = iter->second;
			float* jtj = &jtj_flatten[index * 36];
			for(auto channel = 0; channel < 3; channel++) {
				const float* jacobian_i = encoded_jacobian_i + channel * 6;
				const float* jacobian_j = encoded_jacobian_j + channel * 6;
				for (int jac_row = 0; jac_row < 6; jac_row++) {
					jtj[6 * jac_row + 0] += lambda_smooth_square * jacobian_i[0] * jacobian_j[jac_row];
					jtj[6 * jac_row + 1] += lambda_smooth_square * jacobian_i[1] * jacobian_j[jac_row];
					jtj[6 * jac_row + 2] += lambda_smooth_square * jacobian_i[2] * jacobian_j[jac_row];
					jtj[6 * jac_row + 3] += lambda_smooth_square * jacobian_i[3] * jacobian_j[jac_row];
					jtj[6 * jac_row + 4] += lambda_smooth_square * jacobian_i[4] * jacobian_j[jac_row];
					jtj[6 * jac_row + 5] += lambda_smooth_square * jacobian_i[5] * jacobian_j[jac_row];
				}
			}
		} else {
			//Kill it
			LOG(FATAL) << "The index is not found, something wrong with the index";
		}
		
		//Next fill (node j, node i)
		encoded_jacobian_i = (const float*)(twist_gradient_j);
		encoded_jacobian_j = (const float*)(twist_gradient_i);
		encoded_ij = encode_nodepair(node_j, node_i);
		iter = pair2index.find(encoded_ij);
		if(iter != pair2index.end()) {
			unsigned index = iter->second;
			float* jtj = &jtj_flatten[index * 36];
			for(auto channel = 0; channel < 3; channel++) {
				const float* jacobian_i = encoded_jacobian_i + channel * 6;
				const float* jacobian_j = encoded_jacobian_j + channel * 6;
				for (int jac_row = 0; jac_row < 6; jac_row++) {
					jtj[6 * jac_row + 0] += lambda_smooth_square * jacobian_i[0] * jacobian_j[jac_row];
					jtj[6 * jac_row + 1] += lambda_smooth_square * jacobian_i[1] * jacobian_j[jac_row];
					jtj[6 * jac_row + 2] += lambda_smooth_square * jacobian_i[2] * jacobian_j[jac_row];
					jtj[6 * jac_row + 3] += lambda_smooth_square * jacobian_i[3] * jacobian_j[jac_row];
					jtj[6 * jac_row + 4] += lambda_smooth_square * jacobian_i[4] * jacobian_j[jac_row];
					jtj[6 * jac_row + 5] += lambda_smooth_square * jacobian_i[5] * jacobian_j[jac_row];
				}
			}
		} else {
			//Kill it
			LOG(FATAL) << "The index is not found, something wrong with the index";
		}
	}
}


void surfelwarp::JtJMaterializer::updateFeatureCostJtJBlockHost(std::vector<float> &jtj_flatten) {
	const auto num_nodepairs = m_nodepair2term_map.nodepair_term_range.Size();
	SURFELWARP_CHECK_EQ(num_nodepairs * 36, jtj_flatten.size());
	
	//Prepare the map from node pair to index
	std::map<unsigned, unsigned> pair2index;
	pair2index.clear();
	std::vector<unsigned> encoded_pair_vec;
	m_nodepair2term_map.encoded_nodepair.Download(encoded_pair_vec);
	for(auto i = 0; i < encoded_pair_vec.size(); i++) {
		auto encoded_pair = encoded_pair_vec[i];
		//There should not be duplicate
		auto iter = pair2index.find(encoded_pair);
		assert(iter == pair2index.end());
		
		//insert it
		pair2index.insert(std::make_pair(encoded_pair, i));
	}
	
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
	
	for(auto term_idx = 0; term_idx < knn_arr.size(); term_idx++) {
		float4 target_vertex = depth_vertex_arr[term_idx];
		float4 ref_vertex = reference_vertex_arr[term_idx];
		ushort4 knn = knn_arr[term_idx];
		float4 knn_weight = knn_weight_arr[term_idx];
		TwistGradientOfScalarCost twist_gradient[3];
		device::computePointToPointICPTermJacobian(target_vertex, ref_vertex, knn, knn_weight, node_se3_arr.data(), twist_gradient);
		
		//Fill each knn pair
		unsigned short* knn_ushort = (unsigned short*)(&knn);
		float* weight_float = (float*)(&knn_weight);
		for(auto i = 0; i < 4; i++) {
			unsigned node_i = knn_ushort[i];
			float weight_i = weight_float[i];
			for(auto j = 0; j < 4; j++) {
				if(i == j) continue;
				unsigned node_j = knn_ushort[j];
				float weight_j = weight_float[j];
				auto encoded_ij = encode_nodepair(node_i, node_j);
				auto iter = pair2index.find(encoded_ij);
				if(iter != pair2index.end()) {
					unsigned index = iter->second;
					float* jtj = &jtj_flatten[index * 36];
					for(auto channel = 0; channel < 3; channel++) {
						float* jacobian = (float*)(&twist_gradient[channel]);
						for (int jac_row = 0; jac_row < 6; jac_row++) {
							jtj[6 * jac_row + 0] += lambda_feature_square * weight_i * weight_j * jacobian[0] * jacobian[jac_row];
							jtj[6 * jac_row + 1] += lambda_feature_square * weight_i * weight_j * jacobian[1] * jacobian[jac_row];
							jtj[6 * jac_row + 2] += lambda_feature_square * weight_i * weight_j * jacobian[2] * jacobian[jac_row];
							jtj[6 * jac_row + 3] += lambda_feature_square * weight_i * weight_j * jacobian[3] * jacobian[jac_row];
							jtj[6 * jac_row + 4] += lambda_feature_square * weight_i * weight_j * jacobian[4] * jacobian[jac_row];
							jtj[6 * jac_row + 5] += lambda_feature_square * weight_i * weight_j * jacobian[5] * jacobian[jac_row];
						}
					}
				} else {
					LOG(FATAL) << "The node pair " << node_i << " and " << node_j << " is not found";
				}
			}
		}
	}
}


void surfelwarp::JtJMaterializer::nonDiagonalBlocksSanityCheck() {
	LOG(INFO) << "Sanity check of materialized non-diagonal JtJ block";

	//Compute it at host
	std::vector<float> jtj_blocks;
	const auto num_nodepairs = m_nodepair2term_map.nodepair_term_range.Size();
	jtj_blocks.resize(36 * num_nodepairs);
	memset(jtj_blocks.data(), 0, sizeof(float) * num_nodepairs * 36);
	updateScalarCostJtJBlockHost(jtj_blocks, m_term2jacobian_map.dense_depth_term);
	updateSmoothCostJtJBlockHost(jtj_blocks);
	updateScalarCostJtJBlockHost(jtj_blocks, m_term2jacobian_map.density_map_term, lambda_density_square);
	updateScalarCostJtJBlockHost(jtj_blocks, m_term2jacobian_map.foreground_mask_term, lambda_foreground_square);
	updateFeatureCostJtJBlockHost(jtj_blocks);
	
	//Download the data from device
	std::vector<float> jtj_blocks_dev;
	m_nondiag_blks.ArrayView().Download(jtj_blocks_dev);
	SURFELWARP_CHECK_EQ(jtj_blocks.size(), jtj_blocks_dev.size());
	
	//Compute the error
	auto relative_err = maxRelativeError(jtj_blocks, jtj_blocks_dev, 1e-3f, true);
	/*for(auto i = 0; i < jtj_blocks.size(); i++) {
		auto dev_value = jtj_blocks_dev[i];
		auto host_value = jtj_blocks[i];
		if(std::abs(host_value) > 1e-3) {
			std::cout << "Nonezero elements " << host_value;
		}
	}*/
	LOG(INFO) << "The relative error for non-diagonal jtj blocks is " << relative_err;
}





