#include "common/logging.h"
#include "common/device_intrinsics.h"
#include "core/warp_solver/PenaltyConstants.h"
#include "core/warp_solver/ApplyJtJMatrixFreeHandler.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	enum {
		jtj_dot_blk_size = 6,
		warp_size = 32,
		num_warps = 8,
		thread_blk_size = num_warps * warp_size,
	};

	__device__ __forceinline__ void fillJtJDotXToSharedBlock(
		const float jtj_dot_x[jtj_dot_blk_size],
		float shared_blks[jtj_dot_blk_size][thread_blk_size],
		const float weight_square = 1.0f
	) {
#pragma unroll
		for(auto i = 0; i < jtj_dot_blk_size; i++) {
			shared_blks[i][threadIdx.x] = weight_square * jtj_dot_x[i];
		}
	}

	__device__ __forceinline__ void computeScalarJtJDotX(
		const float* global_x,
		const ScalarCostTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		float jtj_dot_x[jtj_dot_blk_size]
	) {
		const ushort4 term_knn = term2jacobian.knn_array[typed_term_idx];
		const float4 term_knn_weight = term2jacobian.knn_weight_array[typed_term_idx];
		TwistGradientOfScalarCost* twist_gradient = (TwistGradientOfScalarCost*) jtj_dot_x;
		*twist_gradient = term2jacobian.twist_gradient_array[typed_term_idx];
		const unsigned short* term_knn_arr = (const unsigned short*)(&term_knn);
		const float* term_weight_arr = (const float*)(&term_knn_weight);
		
		//Dot with all elements
		float accumlate_dot = 0.0f;
		for(auto i = 0; i < 4; i++) {
			const float* x = global_x + term_knn_arr[i] * jtj_dot_blk_size;
			accumlate_dot += term_weight_arr[i] * twist_gradient->dot(x);
		}

		//Compute with the weight of this elements
		const auto offset = 0 * (node_idx == term_knn.x)
						  + 1 * (node_idx == term_knn.y)
						  + 2 * (node_idx == term_knn.z)
						  + 3 * (node_idx == term_knn.w);
		const float node_weight = term_weight_arr[offset];
#pragma unroll
		for(auto i = 0; i < jtj_dot_blk_size; i++) {
			jtj_dot_x[i] *= (accumlate_dot * node_weight);
		}
	}


	__device__ __forceinline__ void computeSmoothJtJDotX(
		const float* x,
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term,
		float jtj_dot_x[jtj_dot_blk_size]
	) {
		const ushort2 node_ij = term2jacobian.node_graph[typed_term];
		const auto Ti_xj = term2jacobian.Ti_xj[typed_term];
		const auto Tj_xj = term2jacobian.Tj_xj[typed_term];
		const auto validity = term2jacobian.validity_indicator[typed_term];
		TwistGradientOfScalarCost twist_gradient_i[3];
		TwistGradientOfScalarCost twist_gradient_j[3];
		if(validity == 0) {
#pragma unroll
			for(auto i = 0; i < jtj_dot_blk_size; i++)
				jtj_dot_x[i] = 0.0f;
			return;
		}
		computeSmoothTermJacobian(Ti_xj, Tj_xj, twist_gradient_i, twist_gradient_j);

		//Make it flatten
		const unsigned short* node_ij_arr = (const unsigned short*)(&node_ij);
		const bool is_node_i = (node_idx == node_ij.x);
		const float* jacobian = is_node_i ? (const float*)(twist_gradient_i) : (const float*)(twist_gradient_j);

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
		
		//Assign to output
#pragma unroll
		for(auto k = 0; k < jtj_dot_blk_size; k++) {
			jtj_dot_x[k] =   accumlate_dot[0] * jacobian[k]
			               + accumlate_dot[1] * jacobian[6 + k]
			               + accumlate_dot[2] * jacobian[12 + k];
		}
	}


	__device__ __forceinline__ void computeSmoothJtJDotXOnline(
		const float* x,
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term,
		float jtj_dot_x[jtj_dot_blk_size]
	) {
		const ushort2 node_ij = term2jacobian.node_graph[typed_term];
		const auto xi = term2jacobian.reference_node_coords[node_ij.x];
		const auto xj = term2jacobian.reference_node_coords[node_ij.y];
		DualQuaternion dq_i = term2jacobian.node_se3[node_ij.x];
		DualQuaternion dq_j = term2jacobian.node_se3[node_ij.y];
		const mat34 Ti = dq_i.se3_matrix();
		const mat34 Tj = dq_j.se3_matrix();
		const auto validity = term2jacobian.validity_indicator[typed_term];
		TwistGradientOfScalarCost twist_gradient_i[3];
		TwistGradientOfScalarCost twist_gradient_j[3];
		if(validity == 0) {
#pragma unroll
			for(auto i = 0; i < jtj_dot_blk_size; i++)
				jtj_dot_x[i] = 0.0f;
			return;
		}
		computeSmoothTermJacobian(xj, Ti, Tj, twist_gradient_i, twist_gradient_j);

		//Make it flatten
		const unsigned short* node_ij_arr = (const unsigned short*)(&node_ij);
		const bool is_node_i = (node_idx == node_ij.x);
		const float* jacobian = is_node_i ? (const float*)(twist_gradient_i) : (const float*)(twist_gradient_j);

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
		
		//Assign to output
#pragma unroll
		for(auto k = 0; k < jtj_dot_blk_size; k++) {
			jtj_dot_x[k] =   accumlate_dot[0] * jacobian[k]
			               + accumlate_dot[1] * jacobian[6 + k]
			               + accumlate_dot[2] * jacobian[12 + k];
		}
	}


	__device__ __forceinline__ void computeFeatureTermJtJDotX(
		const float* x,
		const Point2PointICPTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		float jtj_dot_x[jtj_dot_blk_size]
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

		//Compute with the weight of this elements
		const auto offset = 0 * (node_idx == knn.x)
						  + 1 * (node_idx == knn.y)
						  + 2 * (node_idx == knn.z)
						  + 3 * (node_idx == knn.w);
		const float node_weight = weight_arr[offset];

		//Assign to output
		const float* jacobian = (const float*)(twist_gradient);
#pragma unroll
		for(auto k = 0; k < jtj_dot_blk_size; k++) {
			jtj_dot_x[k] = node_weight * (
				    accumlate_dot[0] * jacobian[k]
				  + accumlate_dot[1] * jacobian[6 + k]
				  + accumlate_dot[2] * jacobian[12 + k]
			);
		}
	}

	
	__global__ void applyJtJDotWithIndexKernel(
		const DeviceArrayView<float> x,
		const Node2TermsIndex::Node2TermMap node2term,
		const Term2JacobianMaps term2jacobian,
		float* jtj_dot_x,
		const PenaltyConstants constants = PenaltyConstants()
	) {
		const auto node_idx = blockIdx.x;
		const auto term_begin = node2term.offset[node_idx];
		const auto term_end = node2term.offset[node_idx + 1];
		const auto term_size = term_end - term_begin;
		const auto padded_term_size = thread_blk_size * ((term_size + thread_blk_size - 1) / thread_blk_size);
		const auto warp_id = threadIdx.x >> 5;
		const auto lane_id = threadIdx.x & 31;

		//The memory for store the JtResidual result of each threads
		__shared__ float shared_blks[jtj_dot_blk_size][thread_blk_size];
		__shared__ float shared_warp_tmp[num_warps];
		//The memory to perform the reduction
		__shared__ float reduced_blks[jtj_dot_blk_size];
		
		//Zero out the elements
		if(threadIdx.x < jtj_dot_blk_size) reduced_blks[threadIdx.x] = 0.0f;
		__syncthreads();

		//The warp compute terms in the multiple of 32 (the warp size)
		for (auto iter = threadIdx.x; iter < padded_term_size; iter += thread_blk_size)
		{
			//The global term index
			bool term_valid = true;

			//Do computation when the term is inside
			if(iter < term_size)
			{
				//Query the term type
				const auto term_idx = node2term.term_index[term_begin + iter];
				unsigned typed_term_idx;
				TermType term_type;
				query_typed_index(term_idx, node2term.term_offset, term_type, typed_term_idx);

				//Depends on differenct types of terms
				switch (term_type)
				{
				case TermType::DenseImage:
					{
						float term_jtj_dot_x[jtj_dot_blk_size] = {0};
						computeScalarJtJDotX(x, term2jacobian.dense_depth_term, node_idx, typed_term_idx, term_jtj_dot_x);
						fillJtJDotXToSharedBlock(term_jtj_dot_x, shared_blks, constants.DenseDepthSquared());
					}
					break;
				case TermType::Smooth:
					{
						float term_jtj_dot_x[jtj_dot_blk_size] = {0};
						computeSmoothJtJDotX(x, term2jacobian.smooth_term, node_idx, typed_term_idx, term_jtj_dot_x);
						fillJtJDotXToSharedBlock(term_jtj_dot_x, shared_blks, constants.SmoothSquared());
					}
					break;
				/*case TermType::DensityMap:
					{
						float term_jtj_dot_x[jtj_dot_blk_size] = {0};
						computeScalarJtJDotX(x, term2jacobian.density_map_term, node_idx, typed_term_idx, term_jtj_dot_x);
						fillJtJDotXToSharedBlock(term_jtj_dot_x, shared_blks, constants.DensitySquared());
					}
					break;*/
				case TermType::Foreground:
					{
						float term_jtj_dot_x[jtj_dot_blk_size] = {0};
						computeScalarJtJDotX(x, term2jacobian.foreground_mask_term, node_idx, typed_term_idx, term_jtj_dot_x);
						fillJtJDotXToSharedBlock(term_jtj_dot_x, shared_blks, constants.ForegroundSquared());
					}
					break;
				case TermType::Feature:
					{
						float term_jtj_dot_x[jtj_dot_blk_size] = {0};
						computeFeatureTermJtJDotX(x, term2jacobian.sparse_feature_term, node_idx, typed_term_idx, term_jtj_dot_x);
						fillJtJDotXToSharedBlock(term_jtj_dot_x, shared_blks, constants.SparseFeatureSquared());
					}
					break;
				default:
					term_valid = false;
					break;
				}
			}

			//Do a reduction to reduced_men
			for (int i = 0; i < jtj_dot_blk_size; i++) {
				float data = (iter < term_size && term_valid) ? shared_blks[i][threadIdx.x] : 0.0f;
				data = warp_scan(data);
				if (lane_id == warpSize - 1) {
					shared_warp_tmp[warp_id] = data;
				}
				
				__syncthreads();
				data = threadIdx.x < num_warps ? shared_warp_tmp[threadIdx.x] : 0.0f;
				data = warp_scan(data);
				if(threadIdx.x == warpSize - 1) {
					reduced_blks[i] += data;
				}
				__syncthreads();
			}
		} // iterate over terms

		//All the terms that contribute to this value is done, store to global memory
		if(threadIdx.x < jtj_dot_blk_size) jtj_dot_x[jtj_dot_blk_size * node_idx + threadIdx.x] = reduced_blks[threadIdx.x];
	} // the apply jtj kernel

} // namespace device
} // namespace surfelwarp


void surfelwarp::ApplyJtJHandlerMatrixFree::ApplyJtJIndexed(DeviceArrayView<float> x, DeviceArraySlice<float> jtj_dot_x, cudaStream_t stream)
{
	//simple sanity check
	SURFELWARP_CHECK_EQ(x.Size(), jtj_dot_x.Size());
	SURFELWARP_CHECK(x.Size() % 6 == 0);

	//Seems correct
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	dim3 blk(device::thread_blk_size);
	dim3 grid(num_nodes);
	device::applyJtJDotWithIndexKernel<<<grid, blk, 0, stream>>>(
		x, 
		m_node2term_map, 
		m_term2jacobian_map,
		jtj_dot_x.RawPtr(),
		m_penalty_constants
	);
}