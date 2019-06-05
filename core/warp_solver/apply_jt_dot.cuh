#pragma once
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"

namespace surfelwarp { namespace device {
	
	enum {
		jt_dot_blk_size = 6
	};

	__device__ __forceinline__ void computeScalarJacobianTransposeDot(
		const ScalarCostTerm2Jacobian& term2jacobian,
		const float* residual,
		unsigned node_idx, unsigned typed_term_idx,
		float* jt_residual
	) {
		const ushort4 term_knn = term2jacobian.knn_array[typed_term_idx];
		const float residual_value = *residual;
		const auto offset = 0 * (node_idx == term_knn.x)
						  + 1 * (node_idx == term_knn.y)
						  + 2 * (node_idx == term_knn.z)
						  + 3 * (node_idx == term_knn.w);
		const float4* knn_weight = &(term2jacobian.knn_weight_array[typed_term_idx]);
		const float this_weight = ((const float*)knn_weight)[offset];
		TwistGradientOfScalarCost* twist_gradient = (TwistGradientOfScalarCost*) jt_residual;
		*twist_gradient = term2jacobian.twist_gradient_array[typed_term_idx];
#pragma unroll
		for(auto i = 0; i < 6; i++) {
			jt_residual[i] *= (this_weight * residual_value);
		}
	}


	__device__ __forceinline__ void computeSmoothJacobianTransposeDot(
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term,
		const float residual[3],
		float jt_residual[jt_dot_blk_size]
	) {
		const ushort2 node_ij = term2jacobian.node_graph[typed_term];
		const auto xi = term2jacobian.reference_node_coords[node_ij.x];
		const auto xj = term2jacobian.reference_node_coords[node_ij.y];
		DualQuaternion dq_i = term2jacobian.node_se3[node_ij.x];
		DualQuaternion dq_j = term2jacobian.node_se3[node_ij.y];
		const mat34 Ti = dq_i.se3_matrix();
		const mat34 Tj = dq_j.se3_matrix();
		const bool is_node_i = (node_idx == node_ij.x);
		computeSmoothTermJtDot(xj, Ti, Tj, is_node_i, residual, jt_residual);
	}


	__device__ __forceinline__ void computeFeatureJacobianTransposeDot(
		const Point2PointICPTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		const float residual[3],
		float jt_residual[jt_dot_blk_size]
	) {
		const float4 target_vertex = term2jacobian.target_vertex[typed_term_idx];
		//const float4 reference_vertex = term2jacobian.reference_vertex[typed_term_idx];
		const ushort4 knn = term2jacobian.knn[typed_term_idx];
		const float4 knn_weight = term2jacobian.knn_weight[typed_term_idx];
		const float4 warped_vertex = term2jacobian.warped_vertex[typed_term_idx];
		computePoint2PointJtDot(target_vertex, warped_vertex, residual, jt_residual);

		//Multiple with the weight
		const auto offset = 0 * (node_idx == knn.x)
						  + 1 * (node_idx == knn.y)
						  + 2 * (node_idx == knn.z)
						  + 3 * (node_idx == knn.w);
		const float this_weight = ((const float*)(&knn_weight))[offset];
		for(auto i = 0; i < jt_dot_blk_size; i++) {
			//Note that the residual do not need to be augment, only jacobian should be multiplied with weight
			jt_residual[i] *= this_weight;
		}
	}

} // namespace device
} // namespace surfelwarp 