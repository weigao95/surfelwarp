#include "common/device_intrinsics.h"
#include "common/global_configs.h"
#include "core/warp_solver/PenaltyConstants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/apply_jt_dot.cuh"
#include "core/warp_solver/PreconditionerRhsBuilder.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	enum {
		warp_size = 32,
		num_warps = 8,
		thread_blk_size = num_warps * warp_size,
	};

	__device__ __forceinline__ void computeScalarJtResidual(
		const ScalarCostTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		float jt_residual[jt_dot_blk_size]
	) {
		const float* residual = term2jacobian.residual_array.RawPtr() + typed_term_idx;
		computeScalarJacobianTransposeDot(term2jacobian, residual, node_idx, typed_term_idx, jt_residual);
	}

	__device__ __forceinline__ void computeSmoothJtResidual(
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term,
		float jt_residual[jt_dot_blk_size]
	) {
		const ushort2 node_ij = term2jacobian.node_graph[typed_term];
		const auto Ti_xj = term2jacobian.Ti_xj[typed_term];
		const auto Tj_xj = term2jacobian.Tj_xj[typed_term];
		const auto validity = term2jacobian.validity_indicator[typed_term];
		const bool is_node_i = (node_idx == node_ij.x);
		if(validity == 0) {
#pragma unroll
			for(auto i = 0; i < jt_dot_blk_size; i++)
				jt_residual[i] = 0.0f;
			return;
		}
		computeSmoothTermJtResidual(Ti_xj, Tj_xj, is_node_i, jt_residual);
	}


	__device__ __forceinline__ void computeSmoothJtResidualOnline(
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term,
		float jt_residual[jt_dot_blk_size]
	) {
		const ushort2 node_ij = term2jacobian.node_graph[typed_term];
		const auto xi = term2jacobian.reference_node_coords[node_ij.x];
		const auto xj = term2jacobian.reference_node_coords[node_ij.y];
		DualQuaternion dq_i = term2jacobian.node_se3[node_ij.x];
		DualQuaternion dq_j = term2jacobian.node_se3[node_ij.y];
		const auto validity = term2jacobian.validity_indicator[typed_term];
		const mat34 Ti = dq_i.se3_matrix();
		const mat34 Tj = dq_j.se3_matrix();
		const bool is_node_i = (node_idx == node_ij.x);
		if(validity == 0) {
#pragma unroll
			for(auto i = 0; i < jt_dot_blk_size; i++)
				jt_residual[i] = 0.0f;
			return;
		}
		computeSmoothTermJtResidual(xj, Ti, Tj, is_node_i, jt_residual);
	}

	__device__ __forceinline__ void computePoint2PointJtResidual(
		const Point2PointICPTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		float jt_residual[jt_dot_blk_size]
	) {
		const float4 target_vertex = term2jacobian.target_vertex[typed_term_idx];
		//const float4 reference_vertex = term2jacobian.reference_vertex[typed_term_idx];
		const ushort4 knn = term2jacobian.knn[typed_term_idx];
		const float4 knn_weight = term2jacobian.knn_weight[typed_term_idx];
		const float4 warped_vertex = term2jacobian.warped_vertex[typed_term_idx];
		computePoint2PointJtResidual(target_vertex, warped_vertex, jt_residual);

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

	__device__ __forceinline__ void fillScalarJtResidualToSharedBlock(
		const float jt_redisual[jt_dot_blk_size],
		float shared_blks[jt_dot_blk_size][thread_blk_size],
		const float weight_square = 1.0f
	) {
#pragma unroll
		for(auto i = 0; i < jt_dot_blk_size; i++) {
			shared_blks[i][threadIdx.x] = - weight_square * jt_redisual[i];
		}
	}

	__device__ __forceinline__ void incrementScalarJtResidualToSharedBlock(
		const float jt_redisual[jt_dot_blk_size],
		float shared_blks[jt_dot_blk_size][thread_blk_size],
		const float weight_square = 1.0f
	) {
#pragma unroll
		for (auto i = 0; i < jt_dot_blk_size; i++) {
			shared_blks[i][threadIdx.x] += (-weight_square * jt_redisual[i]);
		}
	}
	
	__global__ void computeJtResidualWithIndexKernel(
		const Node2TermsIndex::Node2TermMap node2term,
		const Term2JacobianMaps term2jacobian,
		float* jt_residual,
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

				//Do computation given term_type
				switch (term_type)
				{
				case TermType::DenseImage:
					{
						float term_jt_residual[6] = {0};
						computeScalarJtResidual(term2jacobian.dense_depth_term, node_idx, typed_term_idx, term_jt_residual);
						fillScalarJtResidualToSharedBlock(term_jt_residual, shared_blks, constants.DenseDepthSquared());
#if defined(USE_DENSE_IMAGE_DENSITY_TERM)
						computeScalarJtResidual(term2jacobian.density_map_term, node_idx, typed_term_idx, term_jt_residual);
						incrementScalarJtResidualToSharedBlock(term_jt_residual, shared_blks, constants.DensitySquared());
#endif
					}
					break;
				case TermType::Smooth:
					{
						float term_jt_residual[6] = {0};
						computeSmoothJtResidual(term2jacobian.smooth_term, node_idx, typed_term_idx, term_jt_residual);
						fillScalarJtResidualToSharedBlock(term_jt_residual, shared_blks, constants.SmoothSquared());
					}
					break;
				case TermType::Foreground:
					{
						float term_jt_residual[6] = {0};
						computeScalarJtResidual(term2jacobian.foreground_mask_term, node_idx, typed_term_idx, term_jt_residual);
						fillScalarJtResidualToSharedBlock(term_jt_residual, shared_blks, constants.ForegroundSquared());
					}
					break;
				case TermType::Feature:
					{
						float term_jt_residual[6] = {0};
						computePoint2PointJtResidual(term2jacobian.sparse_feature_term, node_idx, typed_term_idx, term_jt_residual);
						fillScalarJtResidualToSharedBlock(term_jt_residual, shared_blks, constants.SparseFeatureSquared());
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
				//Sync again?
				__syncthreads();
			}
		}

		//All the terms that contribute to this value is done, store to global memory
		if(threadIdx.x < jt_dot_blk_size) jt_residual[jt_dot_blk_size * node_idx + threadIdx.x] = reduced_blks[threadIdx.x];
	}



	__global__ void computeJtResidualWithIndexGlobalIterationKernel(
		const Node2TermsIndex::Node2TermMap node2term,
		const Term2JacobianMaps term2jacobian,
		float* jt_residual,
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
			//The global term index
			bool term_valid = true;

			//Do computation when the term is inside
			if (iter < term_size)
			{
				//Query the term type
				const auto term_idx = node2term.term_index[term_begin + iter];
				unsigned typed_term_idx;
				TermType term_type;
				query_typed_index(term_idx, node2term.term_offset, term_type, typed_term_idx);

				//Do computation given term_type
				switch (term_type)
				{
				case TermType::DenseImage:
				{
					float term_jt_residual[6] = { 0 };
					computeScalarJtResidual(term2jacobian.dense_depth_term, node_idx, typed_term_idx, term_jt_residual);
					fillScalarJtResidualToSharedBlock(term_jt_residual, shared_blks, constants.DenseDepthSquared());
				}
				break;
				case TermType::Smooth:
				{
					float term_jt_residual[6] = { 0 };
					computeSmoothJtResidual(term2jacobian.smooth_term, node_idx, typed_term_idx, term_jt_residual);
					fillScalarJtResidualToSharedBlock(term_jt_residual, shared_blks, constants.SmoothSquared());
				}
				break;
				case TermType::Foreground:
				{
					float term_jt_residual[6] = { 0 };
					computeScalarJtResidual(term2jacobian.foreground_mask_term, node_idx, typed_term_idx, term_jt_residual);
					fillScalarJtResidualToSharedBlock(term_jt_residual, shared_blks, constants.ForegroundSquared());
				}
				break;
				case TermType::Feature:
				{
					float term_jt_residual[6] = { 0 };
					computePoint2PointJtResidual(term2jacobian.sparse_feature_term, node_idx, typed_term_idx, term_jt_residual);
					fillScalarJtResidualToSharedBlock(term_jt_residual, shared_blks, constants.SparseFeatureSquared());
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
				if (threadIdx.x == warpSize - 1) {
					reduced_blks[i] += data;
				}
				//Sync again?
				__syncthreads();
			}
		}

		//All the terms that contribute to this value is done, store to global memory
		if (threadIdx.x < jt_dot_blk_size) jt_residual[jt_dot_blk_size * node_idx + threadIdx.x] = reduced_blks[threadIdx.x];
	}

} // namespace device
} // namespace surfelwarp



//Compute the Jt.dot(residual) using the index from node to term
void surfelwarp::PreconditionerRhsBuilder::ComputeJtResidualIndexed(cudaStream_t stream)
{
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	m_jt_residual.ResizeArrayOrException(num_nodes * device::jt_dot_blk_size);
	dim3 blk(device::thread_blk_size);
	dim3 grid(num_nodes);
	device::computeJtResidualWithIndexKernel<<<grid, blk, 0, stream>>>(
		m_node2term_map,
		m_term2jacobian_map,
		m_jt_residual.Ptr(),
		m_penalty_constants
	);
}


//The interface distingish between the use of local and global interface
void surfelwarp::PreconditionerRhsBuilder::ComputeJtResidualGlobalIteration(cudaStream_t stream) {
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	m_jt_residual.ResizeArrayOrException(num_nodes * device::jt_dot_blk_size);
	dim3 blk(device::thread_blk_size);
	dim3 grid(num_nodes);
	device::computeJtResidualWithIndexGlobalIterationKernel<<<grid, blk, 0, stream>>>(
		m_node2term_map,
		m_term2jacobian_map,
		m_jt_residual.Ptr(),
		m_penalty_constants
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::PreconditionerRhsBuilder::ComputeJtResidualLocalIteration(cudaStream_t stream) {
	ComputeJtResidualIndexed(stream);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


