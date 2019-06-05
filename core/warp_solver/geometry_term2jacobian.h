//
// Created by wei on 4/5/18.
//

#pragma once

#include "common/global_configs.h"
#include "math/vector_ops.hpp"
#include "math/DualQuaternion.hpp"
#include "core/warp_solver/solver_types.h"
#include "geometry_icp_jacobian.cuh"

namespace surfelwarp { namespace device {
	
	__device__ __forceinline__ void computeScalarCostJacobian(
		const ScalarCostTerm2Jacobian& term2jacobian,
		unsigned typed_term_idx,
		TwistGradientOfScalarCost node_jacobians[4]
	) {
		const float4 weight = term2jacobian.knn_weight_array[typed_term_idx];
		const TwistGradientOfScalarCost twist_gradient = term2jacobian.twist_gradient_array[typed_term_idx];
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION)
		node_jacobians[0] = twist_gradient * weight.x;
		node_jacobians[1] = twist_gradient * weight.y;
		node_jacobians[2] = twist_gradient * weight.z;
		node_jacobians[3] = twist_gradient * weight.w;
#else
		const float inv_weight_sum = 1.0f / fabsf_sum(weight);
		node_jacobians[0] = twist_gradient * (weight.x * inv_weight_sum);
		node_jacobians[1] = twist_gradient * (weight.y * inv_weight_sum);
		node_jacobians[2] = twist_gradient * (weight.z * inv_weight_sum);
		node_jacobians[3] = twist_gradient * (weight.w * inv_weight_sum);
#endif
	}


	__device__ __forceinline__ void computeSmoothTermJacobian(
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned smooth_term_index,
		TwistGradientOfScalarCost node_i_jacobian[3], //i is node_graph[term_idx].x
		TwistGradientOfScalarCost node_j_jacobian[3]  //j is node_graph[term_idx].y
	) {
		const auto node_ij = term2jacobian.node_graph[smooth_term_index];
		//const auto xi = term2jacobian.reference_node_coords[node_ij.x]; //Seems not required
		const auto xj = term2jacobian.reference_node_coords[node_ij.y];
		DualQuaternion dq_i = term2jacobian.node_se3[node_ij.x];
		DualQuaternion dq_j = term2jacobian.node_se3[node_ij.y];
		const auto Ti_xj = term2jacobian.Ti_xj[smooth_term_index];
		const auto Tj_xj = term2jacobian.Tj_xj[smooth_term_index];
		const mat34 Ti = dq_i.se3_matrix();
		const mat34 Tj = dq_j.se3_matrix();
		//computeSmoothTermJacobian(xj, Ti, Tj, node_i_jacobian, node_j_jacobian);
		computeSmoothTermJacobian(Ti_xj, Tj_xj, node_i_jacobian, node_j_jacobian);
	}
	

} // namespace device
} // namespace surfelwarp
