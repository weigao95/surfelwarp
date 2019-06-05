#include "core/warp_solver/ApplyJtJMatrixFreeHandler.h"
#include "core/warp_solver/PenaltyConstants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	enum {
		jtj_dot_blk_size = 6,
		warp_size = 32,
	};

	__device__ __forceinline__ void applyScalarJacobianDot(
		const float* x,
		const ScalarCostTerm2Jacobian& term2jacobian,
		unsigned typed_term_idx,
		float* scalar_jacobian_dot_x,
		float term_weight = 1.0f
	) {
		//Load information
		const ushort4 term_knn = term2jacobian.knn_array[typed_term_idx];
		const float4 knn_weight = (term2jacobian.knn_weight_array[typed_term_idx]);
		const TwistGradientOfScalarCost twist_gradient = term2jacobian.twist_gradient_array[typed_term_idx];
		
		//Flatten the 4 values
		const unsigned short* knn_arr = (const unsigned short*)(&term_knn);
		const float* weight_arr = (const float*)(&knn_weight);

		//Perform dot produce
		float accumlate_dot = 0.0f;
		for(auto i = 0; i < 4; i++) {
			const auto node_idx = knn_arr[i];
			const auto weight = weight_arr[i];
			const float* xi = x + (node_idx * jtj_dot_blk_size);
			accumlate_dot += weight * twist_gradient.dot(xi);
		}

		//Save it
		*scalar_jacobian_dot_x = term_weight * accumlate_dot;
	}

	__device__ __forceinline__ void applySmoothJacobianDot(
		const float* x,
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned typed_term_idx,
		float* scalar_jacobian_dot_x,
		const float term_weight
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
		TwistGradientOfScalarCost twist_gradient_i[3];
		TwistGradientOfScalarCost twist_gradient_j[3];
		//computeSmoothTermJacobian(xj, Ti, Tj, twist_gradient_i, twist_gradient_j);
		computeSmoothTermJacobian(Ti_xj, Tj_xj, twist_gradient_i, twist_gradient_j);

		//Make it flatten
		const unsigned short* node_ij_arr = (const unsigned short*)(&node_ij);
		
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

		//Save it
		scalar_jacobian_dot_x[0] = term_weight * accumlate_dot[0];
		scalar_jacobian_dot_x[1] = term_weight * accumlate_dot[1];
		scalar_jacobian_dot_x[2] = term_weight * accumlate_dot[2];
	}

	__device__ __forceinline__ void applyFeatureJacobianDot(
		const float* x,
		const Point2PointICPTerm2Jacobian& term2jacobian,
		unsigned typed_term_idx,
		float* scalar_jacobian_dot_x,
		const float term_weight
	) {
		const float4 target_vertex = term2jacobian.target_vertex[typed_term_idx];
		//const float4 reference_vertex = term2jacobian.reference_vertex[typed_term_idx];
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

		//Save it
		scalar_jacobian_dot_x[0] = term_weight * accumlate_dot[0];
		scalar_jacobian_dot_x[1] = term_weight * accumlate_dot[1];
		scalar_jacobian_dot_x[2] = term_weight * accumlate_dot[2];
	}
	
	__global__ void applyJacobianDotKernel(
		const DeviceArrayView<float> x,
		const Term2JacobianMaps term2jacobian,
		const TermTypeOffset term_offset,
		float* jacobian_dot_x,
		const PenaltyConstants constants = PenaltyConstants()
	) {
		//Parallel over all terms
		const auto term_idx = threadIdx.x + blockDim.x * blockIdx.x;
		
		//Query the term type
		unsigned typed_term_idx, scalar_term_idx;
		TermType term_type;
		query_typed_index(term_idx, term_offset, term_type, typed_term_idx, scalar_term_idx);
		float* scalar_jacobian_dot_x = jacobian_dot_x + scalar_term_idx;

		//Condition on types
		switch (term_type)
		{
		case TermType::DenseImage:
			applyScalarJacobianDot(x, term2jacobian.dense_depth_term, typed_term_idx, scalar_jacobian_dot_x, constants.DenseDepth());
			break;
		case TermType::Smooth:
			applySmoothJacobianDot(x, term2jacobian.smooth_term, typed_term_idx, scalar_jacobian_dot_x, constants.Smooth());
			break;
		/*case TermType::DensityMap:
			applyScalarJacobianDot(x, term2jacobian.density_map_term, typed_term_idx, scalar_jacobian_dot_x, constants.Density());
			break;*/
		case TermType::Foreground:
			applyScalarJacobianDot(x, term2jacobian.foreground_mask_term, typed_term_idx, scalar_jacobian_dot_x, constants.Foreground());
			break;
		case TermType::Feature:
			applyFeatureJacobianDot(x, term2jacobian.sparse_feature_term, typed_term_idx, scalar_jacobian_dot_x, constants.SparseFeature());
			break;
		default:
			break;
		}
	}

} // namespace device
} // namespace surfelwarp

//Assign the static
const unsigned surfelwarp::ApplyJtJHandlerMatrixFree::kMaxNumScalarResidualTerms = 300000;

void surfelwarp::ApplyJtJHandlerMatrixFree::applyJacobianDot(DeviceArrayView<float> x, cudaStream_t stream)
{
	//Resize the array
	const auto term_offset = m_node2term_map.term_offset;
	const auto scalar_term_size = term_offset.ScalarTermSize();
	m_jacobian_dot_x.ResizeArrayOrException(scalar_term_size);

	//Invoke the kernel
	const auto term_size = term_offset.TermSize();
	dim3 blk(256);
	dim3 grid(divUp(term_size, blk.x));
	device::applyJacobianDotKernel<<<grid, blk, 0, stream>>>(
		x,
		m_term2jacobian_map,
		term_offset,
		m_jacobian_dot_x.Ptr(),
		m_penalty_constants
	);
}

void surfelwarp::ApplyJtJHandlerMatrixFree::ApplyJtJSeparate(
	DeviceArrayView<float> x,
	DeviceArraySlice<float> jtj_dot_x,
	cudaStream_t stream
) {
	applyJacobianDot(x, stream);
	applyJacobianTranposeDot(jtj_dot_x, stream);
}