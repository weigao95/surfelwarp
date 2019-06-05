#include "common/device_intrinsics.h"
#include "common/ArraySlice.h"
#include "core/warp_solver/PenaltyConstants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/Node2TermsIndex.h"
#include "core/warp_solver/ApplyJtJMatrixFreeHandler.h"
#include <device_launch_parameters.h>


namespace surfelwarp { namespace device {

	enum {
		jtj_dot_blk_size = 6,
	};

	__device__ __forceinline__ void atomicApplyScalarJtJDot(
		const float* global_x,
		const ScalarCostTerm2Jacobian& term2jacobian,
		unsigned typed_term_idx,
		float* global_jtj_x, //The global output
		float term_weight_square = 1.0f
	) {
		const ushort4 term_knn = term2jacobian.knn_array[typed_term_idx];
		const float4 term_knn_weight = term2jacobian.knn_weight_array[typed_term_idx];
		const TwistGradientOfScalarCost twist_gradient = term2jacobian.twist_gradient_array[typed_term_idx];
		const unsigned short* term_knn_arr = (const unsigned short*)(&term_knn);
		const float* term_weight_arr = (const float*)(&term_knn_weight);
		
		//Dot with all elements
		float accumlate_dot = 0.0f;
		for(auto i = 0; i < 4; i++) {
			const float* x = global_x + term_knn_arr[i] * jtj_dot_blk_size;
			accumlate_dot += term_weight_arr[i] * twist_gradient.dot(x);
		}

		//Apply to the global memory
		const float* jacobian = (const float*)(&twist_gradient);
		for(auto i = 0; i < 4; i++) {
			const auto node_idx = term_knn_arr[i];
			const auto node_weight = term_weight_arr[i];
			float* node_jtj_x = global_jtj_x + jtj_dot_blk_size * node_idx;
			for(auto j = 0; j < jtj_dot_blk_size; j++) {
				const auto value = (term_weight_square * accumlate_dot * node_weight * jacobian[j]);
				atomicAdd(node_jtj_x + j, value);
			}
		}
	}

	__device__ __forceinline__ void atomicApplySmoothJtJDot(
		const float* x,
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned typed_term,
		float* global_jtj_x,
		const float term_weight_square
	) {
		const ushort2 node_ij = term2jacobian.node_graph[typed_term];
		const auto xi = term2jacobian.reference_node_coords[node_ij.x];
		const auto xj = term2jacobian.reference_node_coords[node_ij.y];
		DualQuaternion dq_i = term2jacobian.node_se3[node_ij.x];
		DualQuaternion dq_j = term2jacobian.node_se3[node_ij.y];
		const mat34 Ti = dq_i.se3_matrix();
		const mat34 Tj = dq_j.se3_matrix();
		const auto Ti_xj = term2jacobian.Ti_xj[typed_term];
		const auto Tj_xj = term2jacobian.Tj_xj[typed_term];
		TwistGradientOfScalarCost twist_gradient_i[3];
		TwistGradientOfScalarCost twist_gradient_j[3];
		//computeSmoothTermJacobian(xj, Ti, Tj, twist_gradient_i, twist_gradient_j);
		computeSmoothTermJacobian(Ti_xj, Tj_xj, twist_gradient_i, twist_gradient_j);

		//Make it flatten
		const unsigned short* node_ij_arr = (const unsigned short*)(&node_ij);
		const float* jacobian_i = (const float*)twist_gradient_i;
		const float* jacobian_j = (const float*)twist_gradient_j;

		//Compute the accumlate dot value
		float accumlate_dot[3] = {0};
		for(auto k = 0; k < 2; k++) {
			const auto node_load_from = node_ij_arr[k];
			const float* x_blk_k = x + (node_load_from * 6);
			if(k == 0)
			{
				accumlate_dot[0] += twist_gradient_i[0].dot(x_blk_k);
				accumlate_dot[1] += twist_gradient_i[1].dot(x_blk_k);
				accumlate_dot[2] += twist_gradient_i[2].dot(x_blk_k);
			}
			else
			{
				accumlate_dot[0] += twist_gradient_j[0].dot(x_blk_k);
				accumlate_dot[1] += twist_gradient_j[1].dot(x_blk_k);
				accumlate_dot[2] += twist_gradient_j[2].dot(x_blk_k);
			}
		}

		//Assign to first node
		float* node_jtj_x = global_jtj_x + jtj_dot_blk_size * node_ij.x;
		for(auto k = 0; k < jtj_dot_blk_size; k++) {
			const float value = (accumlate_dot[0] * jacobian_i[k]
			                  +  accumlate_dot[1] * jacobian_i[6 + k]
			                  +  accumlate_dot[2] * jacobian_i[12 + k]) * term_weight_square;
			atomicAdd(node_jtj_x + k, value);
		}

		//Assign to second node
		node_jtj_x = global_jtj_x + jtj_dot_blk_size * node_ij.y;
		for(auto k = 0; k < jtj_dot_blk_size; k++) {
			const float value = (accumlate_dot[0] * jacobian_j[k]
			                  +  accumlate_dot[1] * jacobian_j[6 + k]
			                  +  accumlate_dot[2] * jacobian_j[12 + k]) * term_weight_square;
			atomicAdd(node_jtj_x + k, value);
		}
	}

	__device__ __forceinline__ void atomicApplyFeatureJtJDot(
		const float* x,
		const Point2PointICPTerm2Jacobian& term2jacobian,
		unsigned typed_term_idx,
		float* global_jtj_x,
		const float term_weight_square
	) {
		const float4 target_vertex = term2jacobian.target_vertex[typed_term_idx];
		const float4 reference_vertex = term2jacobian.reference_vertex[typed_term_idx];
		const ushort4 knn = term2jacobian.knn[typed_term_idx];
		const float4 knn_weight = term2jacobian.knn_weight[typed_term_idx];
		const float4 warped_vertex = term2jacobian.warped_vertex[typed_term_idx];
		TwistGradientOfScalarCost twist_gradient[3];
		computePointToPointICPTermJacobian(target_vertex, warped_vertex, twist_gradient);

		//Compute the accumlate dot value
		const unsigned short* knn_arr = (const unsigned short*)(&knn);
		const float* weight_arr = (const float*)(&knn_weight);
		float accumlate_dot[3] = {0};
		for(auto k = 0; k < 4; k++)
		{
			const auto node_load_from = knn_arr[k];
			const float node_weight = weight_arr[k];
			const float* x_blk_k = x + (node_load_from * 6);
			accumlate_dot[0] += node_weight * twist_gradient[0].dot(x_blk_k);
			accumlate_dot[1] += node_weight * twist_gradient[1].dot(x_blk_k);
			accumlate_dot[2] += node_weight * twist_gradient[2].dot(x_blk_k);
		}

		//Assign it
		const float* jacobian = (const float*)(twist_gradient);
		for(auto j = 0; j < 4; j++)
		{
			const float node_weight = weight_arr[j];
			const auto node_idx = knn_arr[j];
			float* node_jtj_x = global_jtj_x + jtj_dot_blk_size * node_idx;
			for(auto k = 0; k < jtj_dot_blk_size; k++) {
				const float value = term_weight_square * node_weight * (accumlate_dot[0] * jacobian[k] + accumlate_dot[1] * jacobian[6 + k] + accumlate_dot[2] * jacobian[12 + k]);
				atomicAdd(node_jtj_x + k, value);
			}
		}
	}


	__global__ void applyJtJDotDirectKernel(
		const DeviceArrayView<float> x,
		const TermTypeOffset term_offset,
		const Term2JacobianMaps term2jacobian,
		float* jtj_dot_x,
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
			atomicApplyScalarJtJDot(x, term2jacobian.dense_depth_term, typed_term_idx, jtj_dot_x, constants.DenseDepthSquared());
			break;
		case TermType::Smooth:
			atomicApplySmoothJtJDot(x, term2jacobian.smooth_term, typed_term_idx, jtj_dot_x, constants.SmoothSquared());
			break;
		/*case TermType::DensityMap:
			atomicApplyScalarJtJDot(x, term2jacobian.density_map_term, typed_term_idx, jtj_dot_x, constants.DensitySquared());
			break;*/
		case TermType::Foreground:
			atomicApplyScalarJtJDot(x, term2jacobian.foreground_mask_term, typed_term_idx, jtj_dot_x, constants.ForegroundSquared());
			break;
		case TermType::Feature:
			atomicApplyFeatureJtJDot(x, term2jacobian.sparse_feature_term, typed_term_idx, jtj_dot_x, constants.SparseFeatureSquared());
			break;
		default:
			break;
		}
	}


} // namespace device
} // namespace surfelwarp



void surfelwarp::ApplyJtJHandlerMatrixFree::ApplyJtJAtomic(
	DeviceArrayView<float> x,
	DeviceArraySlice<float> jtj_dot_x,
	cudaStream_t stream
) {
	//Simple sanity check
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	SURFELWARP_CHECK_EQ(x.Size(), device::jtj_dot_blk_size * num_nodes);
	SURFELWARP_CHECK_EQ(jtj_dot_x.Size(), device::jtj_dot_blk_size * num_nodes);
	
	//Zero out the memory
	cudaSafeCall(cudaMemsetAsync(jtj_dot_x.RawPtr(), 0, sizeof(float) * jtj_dot_x.Size(), stream));
	
	//Perform Jt.dot(residual)
	const auto term_offset = m_node2term_map.term_offset;
	const unsigned term_size = term_offset.TermSize();
	dim3 blk(128);
	dim3 grid(divUp(term_size, blk.x));
	device::applyJtJDotDirectKernel<<<grid, blk, 0, stream>>>(
		x,
		term_offset,
		m_term2jacobian_map,
		jtj_dot_x,
		m_penalty_constants
	);
}