#include "common/device_intrinsics.h"
#include "core/warp_solver/PenaltyConstants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/PreconditionerRhsBuilder.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	enum {
		jt_residual_blk_size = 6,
	};

	__device__ __forceinline__ void atomicApplyJtResidual(
		const ScalarCostTerm2Jacobian& term2jacobian,
		unsigned typed_term_idx,
		float* global_jt_residual, //The global output
		float term_weight_square = 1.0f
	) {
		//Load information
		const ushort4 term_knn = term2jacobian.knn_array[typed_term_idx];
		const float residual = term2jacobian.residual_array[typed_term_idx];
		const float4 knn_weight = (term2jacobian.knn_weight_array[typed_term_idx]);
		const TwistGradientOfScalarCost twist_gradient = term2jacobian.twist_gradient_array[typed_term_idx];
		
		//Flatten the 4 values
		const float* twist_jacobian = (const float*)(&twist_gradient);
		const unsigned short* knn_arr = (const unsigned short*)(&term_knn);
		const float* weight_arr = (const float*)(&knn_weight);

		//Apply it globally using atomicAdd
		for(auto i = 0; i < 4; i++) {
			const auto node_idx = knn_arr[i];
			const auto weight = weight_arr[i];
			float* node_jt_residual = &(global_jt_residual[jt_residual_blk_size * node_idx]);
			for(auto j = 0; j < jt_residual_blk_size; j++) {
				const float value = residual * term_weight_square * weight * twist_jacobian[j];
				atomicAdd(node_jt_residual + j, -value);
			}
		}
	}

	__device__ __forceinline__ void atomicApplyJtResidual(
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned typed_term_idx,
		float* global_jt_residual,
		const float term_weight_square
	) {
		const ushort2 node_ij = term2jacobian.node_graph[typed_term_idx];
		const auto xi = term2jacobian.reference_node_coords[node_ij.x];
		const auto xj = term2jacobian.reference_node_coords[node_ij.y];
		DualQuaternion dq_i = term2jacobian.node_se3[node_ij.x];
		DualQuaternion dq_j = term2jacobian.node_se3[node_ij.y];
		const mat34 Ti = dq_i.se3_matrix();
		const mat34 Tj = dq_j.se3_matrix();
		const auto Ti_xj = term2jacobian.Ti_xj[typed_term_idx];
		const auto Tj_xj = term2jacobian.Tj_xj[typed_term_idx];
		float jt_residual[6];

		//For node i
		//computeSmoothTermJtResidual(xj, Ti, Tj, true, jt_residual);
		computeSmoothTermJtResidual(Ti_xj, Tj_xj, true, jt_residual);
		float* node_jt_residual = &(global_jt_residual[jt_residual_blk_size * node_ij.x]);
#pragma unroll
		for(auto j = 0; j < jt_residual_blk_size; j++) {
			const float value = term_weight_square * jt_residual[j];
			atomicAdd(node_jt_residual + j, -value);
		}

		//For node j
		//computeSmoothTermJtResidual(xj, Ti, Tj, false, jt_residual);
		computeSmoothTermJtResidual(Ti_xj, Tj_xj, false, jt_residual);
		node_jt_residual = &(global_jt_residual[jt_residual_blk_size * node_ij.y]);
#pragma unroll
		for(auto j = 0; j < jt_residual_blk_size; j++) {
			const float value = term_weight_square * jt_residual[j];
			atomicAdd(node_jt_residual + j, -value);
		}
	}


	__device__ __forceinline__ void atomicApplyJtResidual(
		const Point2PointICPTerm2Jacobian& term2jacobian,
		unsigned typed_term_idx,
		float* global_jt_residual,
		float term_weight_square
	) {
		const float4 target_vertex = term2jacobian.target_vertex[typed_term_idx];
		//const float4 reference_vertex = term2jacobian.reference_vertex[typed_term_idx];
		const ushort4 knn = term2jacobian.knn[typed_term_idx];
		const float4 knn_weight = term2jacobian.knn_weight[typed_term_idx];
		const float4 warped_vertex = term2jacobian.warped_vertex[typed_term_idx];
		float jt_residual[6];
		computePoint2PointJtResidual(target_vertex, warped_vertex, jt_residual);

		//Flatten the 4 values
		const unsigned short* knn_arr = (const unsigned short*)(&knn);
		const float* weight_arr = (const float*)(&knn_weight);

		//Apply it globally using atomicAdd
		for(auto i = 0; i < 4; i++) {
			const auto node_idx = knn_arr[i];
			const auto weight = weight_arr[i];
			float* node_jt_residual = &(global_jt_residual[jt_residual_blk_size * node_idx]);
			for(auto j = 0; j < jt_residual_blk_size; j++) {
				const float value = term_weight_square * weight * jt_residual[j];
				atomicAdd(node_jt_residual + j, -value);
			}
		}
	}

	__global__ void computeJtResidualDirectKernel(
		const Term2JacobianMaps term2jacobian,
		const TermTypeOffset term_offset,
		float* jt_residual,
		const PenaltyConstants constants = PenaltyConstants()
	) {
		//Parallel over all terms
		const auto term_idx = threadIdx.x + blockDim.x * blockIdx.x;
		
		//Query the term type
		unsigned typed_term_idx;
		TermType term_type;
		query_typed_index(term_idx, term_offset, term_type, typed_term_idx);

		//Condition on types
		switch (term_type)
		{
		case TermType::DenseImage:
			atomicApplyJtResidual(term2jacobian.dense_depth_term, typed_term_idx, jt_residual, constants.DenseDepthSquared());
			break;
		case TermType::Smooth:
			atomicApplyJtResidual(term2jacobian.smooth_term, typed_term_idx, jt_residual, constants.SmoothSquared());
			break;
		/*case TermType::DensityMap:
			atomicApplyJtResidual(term2jacobian.density_map_term, typed_term_idx, jt_residual, constants.DensitySquared());
			break;*/
		case TermType::Foreground:
			atomicApplyJtResidual(term2jacobian.foreground_mask_term, typed_term_idx, jt_residual, constants.ForegroundSquared());
			break;
		case TermType::Feature:
			atomicApplyJtResidual(term2jacobian.sparse_feature_term, typed_term_idx, jt_residual, constants.SparseFeatureSquared());
			break;
		default:
			break;
		}
	}


} // namespace device
} // namespace surfelwarp

//Compute Jt.dot(residual) directly using atomicAdd
void surfelwarp::PreconditionerRhsBuilder::ComputeJtResidualAtomic(cudaStream_t stream) {
	//First, zero out all the elements
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	m_jt_residual.ResizeArrayOrException(num_nodes * device::jt_residual_blk_size);
	cudaSafeCall(cudaMemsetAsync(m_jt_residual.Ptr(), 0, m_jt_residual.ArraySize() * sizeof(float), stream));

	//Perform Jt.dot(residual)
	const auto term_offset = m_node2term_map.term_offset;
	const unsigned term_size = term_offset.TermSize();
	dim3 blk(128);
	dim3 grid(divUp(term_size, blk.x));
	device::computeJtResidualDirectKernel<<<grid, blk, 0, stream>>>(
		m_term2jacobian_map,
		term_offset,
		m_jt_residual.Ptr(),
		m_penalty_constants
	);
}