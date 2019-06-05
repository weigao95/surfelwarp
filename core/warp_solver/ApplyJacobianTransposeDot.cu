#include "common/device_intrinsics.h"
#include "core/warp_solver/ApplyJtJMatrixFreeHandler.h"
#include "core/warp_solver/PenaltyConstants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/apply_jt_dot.cuh"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	enum {
		warp_size = 32,
		num_warps = 4,
		thread_blk_size = num_warps * warp_size,
	};

	__device__ __forceinline__ void fillScalarJtXToSharedBlock(
		const float jt_redisual[jt_dot_blk_size],
		float shared_blks[jt_dot_blk_size][thread_blk_size],
		const float weight_square = 1.0f
	) {
#pragma unroll
		for(auto i = 0; i < jt_dot_blk_size; i++) {
			shared_blks[i][threadIdx.x] = weight_square * jt_redisual[i];
		}
	}


	__global__ void applyJacobianTransposeDotKernel(
		const float* residual,
		const Node2TermsIndex::Node2TermMap node2term,
		const Term2JacobianMaps term2jacobian,
		float* jt_dot_x,
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
		__shared__ float shared_blks[jt_dot_blk_size][thread_blk_size];
		__shared__ float shared_warp_tmp[num_warps];
		//The memory to perform the reduction
		__shared__ float reduced_blks[jt_dot_blk_size];
#pragma unroll
		for (auto iter = threadIdx.x; iter < jt_dot_blk_size; iter += thread_blk_size) {
			reduced_blks[iter] = 0.0f;
		}
		__syncthreads();

		//The warp compute terms in the multiple of 32 (the warp size)
		for (auto iter = threadIdx.x; iter < padded_term_size; iter += thread_blk_size)
		{
			//The valid indicator
			bool term_valid = true;

			//Do computation when the term is inside
			if(iter < term_size)
			{
				//Query the term type
				const auto term_idx = node2term.term_index[term_begin + iter];
				unsigned typed_term_idx, scalar_term_idx;
				TermType term_type;
				query_typed_index(term_idx, node2term.term_offset, term_type, typed_term_idx, scalar_term_idx);
				const float* term_residual = residual + scalar_term_idx;

				//Do computation given term_type
				switch (term_type)
				{
				case TermType::DenseImage:
					{
						float term_jt_residual[6] = {0};
						computeScalarJacobianTransposeDot(term2jacobian.dense_depth_term, term_residual, node_idx, typed_term_idx, term_jt_residual);
						fillScalarJtXToSharedBlock(term_jt_residual, shared_blks, constants.DenseDepth());
					}
					break;
				case TermType::Smooth:
					{
						float term_jt_residual[6] = {0};
						computeSmoothJacobianTransposeDot(term2jacobian.smooth_term, node_idx, typed_term_idx, term_residual, term_jt_residual);
						fillScalarJtXToSharedBlock(term_jt_residual, shared_blks, constants.Smooth());
					}
					break;
				/*case TermType::DensityMap:
					{
						float term_jt_residual[6] = {0};
						computeScalarJacobianTransposeDot(term2jacobian.density_map_term, term_residual, node_idx, typed_term_idx, term_jt_residual);
						fillScalarJtXToSharedBlock(term_jt_residual, shared_blks, constants.Density());
					}
					break;*/
				case TermType::Foreground:
					{
						float term_jt_residual[6] = {0};
						computeScalarJacobianTransposeDot(term2jacobian.foreground_mask_term, term_residual, node_idx, typed_term_idx, term_jt_residual);
						fillScalarJtXToSharedBlock(term_jt_residual, shared_blks, constants.Foreground());
					}
					break;
				case TermType::Feature:
					{
						float term_jt_residual[6] = {0};
						computeFeatureJacobianTransposeDot(term2jacobian.sparse_feature_term, node_idx, typed_term_idx, term_residual, term_jt_residual);
						fillScalarJtXToSharedBlock(term_jt_residual, shared_blks, constants.SparseFeature());
					}
					break;
				default:
					term_valid = false;
					break;
				}
			}

			//Do a reduction to reduced_men
			__syncthreads();
			for (int i = 0; i < jt_dot_blk_size; i++) {
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
		}

		//All the terms that contribute to this value is done, store to global memory
		if(threadIdx.x < jt_dot_blk_size) jt_dot_x[jt_dot_blk_size * node_idx + threadIdx.x] = reduced_blks[threadIdx.x];
	}


} // namespace device
} // namespace surfelwarp


void surfelwarp::ApplyJtJHandlerMatrixFree::applyJacobianTranposeDot(DeviceArraySlice<float> jtj_dot_x, cudaStream_t stream)
{
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	SURFELWARP_CHECK_EQ(jtj_dot_x.Size(), device::jt_dot_blk_size * num_nodes);
	dim3 blk(device::thread_blk_size);
	dim3 grid(num_nodes);
	device::applyJacobianTransposeDotKernel<<<grid, blk, 0, stream>>>(
		m_jacobian_dot_x.Ptr(),
		m_node2term_map, 
		m_term2jacobian_map, 
		jtj_dot_x.RawPtr(),
		m_penalty_constants
	);
}