#pragma once
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/solver_encode.h"

namespace surfelwarp { namespace device {
	
	__device__ __forceinline__ void computeScalarJtJBlockJacobian(
		const ScalarCostTerm2Jacobian& term2jacobian,
		unsigned encoded_pair, unsigned typed_term_idx,
		float jacobian[6],
		float* weight
	) {
		const ushort4 term_knn = term2jacobian.knn_array[typed_term_idx];
		unsigned node_i, node_j;
		decode_nodepair(encoded_pair, node_i, node_j);
		const auto offset_i = 0 * (node_i == term_knn.x)
						    + 1 * (node_i == term_knn.y)
						    + 2 * (node_i == term_knn.z)
						    + 3 * (node_i == term_knn.w);
		const auto offset_j = 0 * (node_j == term_knn.x)
						    + 1 * (node_j == term_knn.y)
						    + 2 * (node_j == term_knn.z)
						    + 3 * (node_j == term_knn.w);
		const float4* knn_weight = &(term2jacobian.knn_weight_array[typed_term_idx]);
		const float node_i_weight = ((const float*)knn_weight)[offset_i];
		const float node_j_weight = ((const float*)knn_weight)[offset_j];
		TwistGradientOfScalarCost* twist_gradient = (TwistGradientOfScalarCost*) jacobian;
		*twist_gradient = term2jacobian.twist_gradient_array[typed_term_idx];
		*weight = node_i_weight * node_j_weight;
	}




	__device__ __forceinline__ void computeFeatureJtJBlockJacobian(
		const Point2PointICPTerm2Jacobian& term2jacobian,
		unsigned encoded_pair, unsigned typed_term_idx,
		float* channelled_jacobian,
		float* weight
	) {
		const float4 target_vertex = term2jacobian.target_vertex[typed_term_idx];
		//const float4 reference_vertex = term2jacobian.reference_vertex[typed_term_idx];
		const ushort4 knn = term2jacobian.knn[typed_term_idx];
		const float4 knn_weight = term2jacobian.knn_weight[typed_term_idx];
		const float4 warped_vertex = term2jacobian.warped_vertex[typed_term_idx];

		//Compute the weight
		unsigned node_i, node_j;
		decode_nodepair(encoded_pair, node_i, node_j);
		const auto offset_i = 0 * (node_i == knn.x)
						    + 1 * (node_i == knn.y)
						    + 2 * (node_i == knn.z)
						    + 3 * (node_i == knn.w);
		const auto offset_j = 0 * (node_j == knn.x)
						    + 1 * (node_j == knn.y)
						    + 2 * (node_j == knn.z)
						    + 3 * (node_j == knn.w);
		const float node_i_weight = ((const float*)(&knn_weight))[offset_i];
		const float node_j_weight = ((const float*)(&knn_weight))[offset_j];
		*weight = (node_i_weight * node_j_weight);

		//Compute the jacobian
		computePointToPointICPTermJacobian(target_vertex, warped_vertex, (TwistGradientOfScalarCost*)channelled_jacobian);
	}
} // namespace device
} // namespace surfelwarp