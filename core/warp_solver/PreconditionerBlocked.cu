#include "common/device_intrinsics.h"
#include "common/global_configs.h"
#include "core/warp_solver/PreconditionerRhsBuilder.h"
#include "core/warp_solver/Node2TermsIndex.h"
#include "core/warp_solver/term_offset_types.h"
#include "core/warp_solver/geometry_term2jacobian.h"
#include "core/warp_solver/PenaltyConstants.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	enum {
		preconditioner_blk_size = 36,
		warp_size = 32,
	};

	__device__ __forceinline__ void computeJtJDiagonalJacobian(
		const ScalarCostTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		float* jacobian
	) {
		const ushort4 term_knn = term2jacobian.knn_array[typed_term_idx];
		const auto offset = 0 * (node_idx == term_knn.x)
						  + 1 * (node_idx == term_knn.y)
						  + 2 * (node_idx == term_knn.z)
						  + 3 * (node_idx == term_knn.w);
		const float4* knn_weight = &(term2jacobian.knn_weight_array[typed_term_idx]);
		const float this_weight = ((const float*)knn_weight)[offset];
		TwistGradientOfScalarCost* twist_gradient = (TwistGradientOfScalarCost*) jacobian;
		*twist_gradient = term2jacobian.twist_gradient_array[typed_term_idx];
#pragma unroll
		for(auto i = 0; i < 6; i++) {
			jacobian[i] *= this_weight;
		}
	}

	__device__ __forceinline__ void computeJtJDiagonalJacobian(
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term,
		float* channelled_jacobian
	) {
		const ushort2 node_ij = term2jacobian.node_graph[typed_term];
		const auto Ti_xj = term2jacobian.Ti_xj[typed_term];
		const auto Tj_xj = term2jacobian.Tj_xj[typed_term];
		const auto validity = term2jacobian.validity_indicator[typed_term];
		const bool is_node_i = (node_idx == node_ij.x);
		if(validity == 0) {
#pragma unroll
			for(auto i = 0; i < 18; i++)
				channelled_jacobian[i] = 0.0f;
			return;
		}
		computeSmoothTermJacobian(Ti_xj, Tj_xj, is_node_i, (TwistGradientOfScalarCost*)channelled_jacobian);
	}


	__device__ __forceinline__ void computeJtJDiagonalJacobianOnline(
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term,
		float* channelled_jacobian
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
			for(auto i = 0; i < 18; i++)
				channelled_jacobian[i] = 0.0f;
			return;
		}
		computeSmoothTermJacobian(xj, Ti, Tj, is_node_i, (TwistGradientOfScalarCost*)channelled_jacobian);
	}
	

	__device__ __forceinline__ void computeJtJDiagonalJacobian(
		const Point2PointICPTerm2Jacobian& term2jacobian,
		unsigned node_idx, unsigned typed_term_idx,
		float* channelled_jacobian
	) {
		const float4 target_vertex = term2jacobian.target_vertex[typed_term_idx];
		//const float4 reference_vertex = term2jacobian.reference_vertex[typed_term_idx];
		const ushort4 knn = term2jacobian.knn[typed_term_idx];
		const float4 knn_weight = term2jacobian.knn_weight[typed_term_idx];
		const float4 warped_vertex = term2jacobian.warped_vertex[typed_term_idx];
		computePointToPointICPTermJacobian(target_vertex, warped_vertex, (TwistGradientOfScalarCost*)channelled_jacobian);

		//Multiple with the weight
		const auto offset = 0 * (node_idx == knn.x)
						  + 1 * (node_idx == knn.y)
						  + 2 * (node_idx == knn.z)
						  + 3 * (node_idx == knn.w);
		const float this_weight = ((const float*)(&knn_weight))[offset];
#pragma unroll
		for(auto i = 0; i < 18; i++) {
			channelled_jacobian[i] *= this_weight;
		}
	}

	__device__ __forceinline__ void fillScalarJtJToSharedBlock(
		const float jacobian[6],
		float shared_jtj_blks[preconditioner_blk_size][warp_size],
		const float weight_square = 1.0f
	) {
#pragma unroll
		for (int jac_row = 0; jac_row < 6; jac_row++) {
			shared_jtj_blks[6 * jac_row + 0][threadIdx.x] = weight_square * jacobian[0] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 1][threadIdx.x] = weight_square * jacobian[1] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 2][threadIdx.x] = weight_square * jacobian[2] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 3][threadIdx.x] = weight_square * jacobian[3] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 4][threadIdx.x] = weight_square * jacobian[4] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 5][threadIdx.x] = weight_square * jacobian[5] * jacobian[jac_row];
		}
	}

	__device__ __forceinline__ void incrementScalarJtJToSharedBlock(
		const float jacobian[6],
		float shared_jtj_blks[preconditioner_blk_size][warp_size],
		const float weight_square = 1.0f
	) {
#pragma unroll
		for (int jac_row = 0; jac_row < 6; jac_row++) {
			shared_jtj_blks[6 * jac_row + 0][threadIdx.x] += weight_square * jacobian[0] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 1][threadIdx.x] += weight_square * jacobian[1] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 2][threadIdx.x] += weight_square * jacobian[2] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 3][threadIdx.x] += weight_square * jacobian[3] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 4][threadIdx.x] += weight_square * jacobian[4] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 5][threadIdx.x] += weight_square * jacobian[5] * jacobian[jac_row];
		}
	}

	__device__ __forceinline__ void fillChannelledJtJToSharedBlock(
		const float jacobian_channelled[18],
		float shared_jtj_blks[preconditioner_blk_size][warp_size],
		const float weight_square = 1.0f
	) {
		//The first iteration: assign
		const float* jacobian = jacobian_channelled;
#pragma unroll
		for (int jac_row = 0; jac_row < 6; jac_row++) {
			shared_jtj_blks[6 * jac_row + 0][threadIdx.x] = weight_square * jacobian[0] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 1][threadIdx.x] = weight_square * jacobian[1] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 2][threadIdx.x] = weight_square * jacobian[2] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 3][threadIdx.x] = weight_square * jacobian[3] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 4][threadIdx.x] = weight_square * jacobian[4] * jacobian[jac_row];
			shared_jtj_blks[6 * jac_row + 5][threadIdx.x] = weight_square * jacobian[5] * jacobian[jac_row];
		}
		

		//The next 2 iterations: plus
		for(auto channel = 1; channel < 3; channel++) {
			jacobian = &(jacobian_channelled[channel * 6]);
#pragma unroll
			for (int jac_row = 0; jac_row < 6; jac_row++) {
				shared_jtj_blks[6 * jac_row + 0][threadIdx.x] += weight_square * jacobian[0] * jacobian[jac_row];
				shared_jtj_blks[6 * jac_row + 1][threadIdx.x] += weight_square * jacobian[1] * jacobian[jac_row];
				shared_jtj_blks[6 * jac_row + 2][threadIdx.x] += weight_square * jacobian[2] * jacobian[jac_row];
				shared_jtj_blks[6 * jac_row + 3][threadIdx.x] += weight_square * jacobian[3] * jacobian[jac_row];
				shared_jtj_blks[6 * jac_row + 4][threadIdx.x] += weight_square * jacobian[4] * jacobian[jac_row];
				shared_jtj_blks[6 * jac_row + 5][threadIdx.x] += weight_square * jacobian[5] * jacobian[jac_row];
			}
		}
	}
	

	__global__ void computeBlockDiagonalPreconditionerKernel(
		const Node2TermsIndex::Node2TermMap node2term,
		const Term2JacobianMaps term2jacobian,
		float* diagonal_preconditioner,
		const PenaltyConstants constants = PenaltyConstants()
	) {
		const auto node_idx = blockIdx.x;
		const auto term_begin = node2term.offset[node_idx];
		const auto term_end = node2term.offset[node_idx + 1];
		const auto term_size = term_end - term_begin;
		const auto padded_term_size = warp_size * ((term_size + warp_size - 1) / warp_size);

		//The memory for store the JtJ result of each threads
		__shared__ float shared_blks[preconditioner_blk_size][warp_size];
		//The memory to perform the reduction
		__shared__ float reduced_blks[preconditioner_blk_size];
#pragma unroll
		for (auto iter = threadIdx.x; iter < preconditioner_blk_size; iter += warp_size) {
			reduced_blks[iter] = 0.0f;
		}
		__syncthreads();


		//The warp compute terms in the multiple of 32 (the warp size)
		for (auto iter = threadIdx.x; iter < padded_term_size; iter += warp_size)
		{
			//The global term index
			bool term_valid = true;
			
			// Do computation when the term is inside
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
						float jacobian[6] = {0};
						computeJtJDiagonalJacobian(term2jacobian.dense_depth_term, node_idx, typed_term_idx, jacobian);
						fillScalarJtJToSharedBlock(jacobian, shared_blks, constants.DenseDepthSquared());
#if defined(USE_DENSE_IMAGE_DENSITY_TERM)
						computeJtJDiagonalJacobian(term2jacobian.density_map_term, node_idx, typed_term_idx, jacobian);
						incrementScalarJtJToSharedBlock(jacobian, shared_blks, constants.DensitySquared());
#endif
					}
					break;
				case TermType::Smooth:
					{
						float channelled_jacobian[18] = {0};
						computeJtJDiagonalJacobian(term2jacobian.smooth_term, node_idx, typed_term_idx, channelled_jacobian);
						fillChannelledJtJToSharedBlock(channelled_jacobian, shared_blks, constants.SmoothSquared());
					}
					break;
				case TermType::Foreground:
					{
						float jacobian[6] = {0};
						computeJtJDiagonalJacobian(term2jacobian.foreground_mask_term, node_idx, typed_term_idx, jacobian);
						fillScalarJtJToSharedBlock(jacobian, shared_blks, constants.ForegroundSquared());
					}
					break;
				case TermType::Feature:
					{
						float channelled_jacobian[18] = {0};
						computeJtJDiagonalJacobian(term2jacobian.sparse_feature_term, node_idx, typed_term_idx, channelled_jacobian);
						fillChannelledJtJToSharedBlock(channelled_jacobian, shared_blks, constants.SparseFeatureSquared());
					}
					break;
				case TermType::Invalid:
					term_valid = false;
					break;
				} // the switch of types
			}
			
			
			//Do a reduction to reduced_men
			__syncthreads();
			for (int i = 0; i < preconditioner_blk_size; i++) {
				float data = (iter < term_size && term_valid) ? shared_blks[i][threadIdx.x] : 0.0f;
				data = warp_scan(data);
				if (threadIdx.x == warpSize - 1) {
					reduced_blks[i] += data;
				}
				
				//Another sync here for reduced mem?
				//__syncthreads();
			}
		}
		
		
		// add small offset to diagonal elements
		if (threadIdx.x < 6) {
			reduced_blks[7*threadIdx.x] += 1e-2f;
		}
		
		__syncthreads();

		//All the terms that contribute to this value is done, store to global memory
#pragma unroll
		for (int i = threadIdx.x; i < preconditioner_blk_size; i += 32) {
			diagonal_preconditioner[preconditioner_blk_size * node_idx + i] = reduced_blks[i];
		}
	}



	__global__ void computeBlockDiagonalPreconditionerGlobalIterationKernel(
		const Node2TermsIndex::Node2TermMap node2term,
		const Term2JacobianMaps term2jacobian,
		float* diagonal_preconditioner,
		const PenaltyConstants constants = PenaltyConstants()
	) {
		const auto node_idx = blockIdx.x;
		const auto term_begin = node2term.offset[node_idx];
		const auto term_end = node2term.offset[node_idx + 1];
		const auto term_size = term_end - term_begin;
		const auto padded_term_size = warp_size * ((term_size + warp_size - 1) / warp_size);

		//The memory for store the JtJ result of each threads
		__shared__ float shared_blks[preconditioner_blk_size][warp_size];
		//The memory to perform the reduction
		__shared__ float reduced_blks[preconditioner_blk_size];
#pragma unroll
		for (auto iter = threadIdx.x; iter < preconditioner_blk_size; iter += warp_size) {
			reduced_blks[iter] = 0.0f;
		}
		__syncthreads();


		//The warp compute terms in the multiple of 32 (the warp size)
		for (auto iter = threadIdx.x; iter < padded_term_size; iter += warp_size)
		{
			//The global term index
			bool term_valid = true;

			// Do computation when the term is inside
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
					float jacobian[6] = { 0 };
					computeJtJDiagonalJacobian(term2jacobian.dense_depth_term, node_idx, typed_term_idx, jacobian);
					fillScalarJtJToSharedBlock(jacobian, shared_blks, constants.DenseDepthSquared());
				}
				break;
				case TermType::Smooth:
				{
					float channelled_jacobian[18] = { 0 };
					computeJtJDiagonalJacobian(term2jacobian.smooth_term, node_idx, typed_term_idx, channelled_jacobian);
					fillChannelledJtJToSharedBlock(channelled_jacobian, shared_blks, constants.SmoothSquared());
				}
				break;
				case TermType::Foreground:
				{
					float jacobian[6] = { 0 };
					computeJtJDiagonalJacobian(term2jacobian.foreground_mask_term, node_idx, typed_term_idx, jacobian);
					fillScalarJtJToSharedBlock(jacobian, shared_blks, constants.ForegroundSquared());
				}
				break;
				case TermType::Feature:
				{
					float channelled_jacobian[18] = { 0 };
					computeJtJDiagonalJacobian(term2jacobian.sparse_feature_term, node_idx, typed_term_idx, channelled_jacobian);
					fillChannelledJtJToSharedBlock(channelled_jacobian, shared_blks, constants.SparseFeatureSquared());
				}
				break;
				default:
					term_valid = false;
					break;
				} // the switch of types
			}

			//Continue if everything is outside
			if(__all(!term_valid))
				continue;

			//Do a reduction to reduced_men
			__syncthreads();
			for (int i = 0; i < preconditioner_blk_size; i++) {
				float data = (iter < term_size && term_valid) ? shared_blks[i][threadIdx.x] : 0.0f;
				data = warp_scan(data);
				if (threadIdx.x == warpSize - 1) {
					reduced_blks[i] += data;
				}

				//Another sync here for reduced mem?
				//__syncthreads();
			}
		} // The computing loop


		// add small offset to diagonal elements
		if (threadIdx.x < 6) {
			reduced_blks[7 * threadIdx.x] += 1e-2f;
		}

		__syncthreads();

		//All the terms that contribute to this value is done, store to global memory
#pragma unroll
		for (int i = threadIdx.x; i < preconditioner_blk_size; i += 32) {
			diagonal_preconditioner[preconditioner_blk_size * node_idx + i] = reduced_blks[i];
		}
	}

} // namespace device
} // namespace surfelwarp


void surfelwarp::PreconditionerRhsBuilder::ComputeDiagonalBlocks(cudaStream_t stream) {
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	m_block_preconditioner.ResizeArrayOrException(num_nodes * device::preconditioner_blk_size);
	dim3 blk(device::warp_size);
	dim3 grid(num_nodes);
	device::computeBlockDiagonalPreconditionerKernel<<<grid, blk, 0, stream>>>(
		m_node2term_map,
		m_term2jacobian_map,
		m_block_preconditioner.Ptr(),
		m_penalty_constants
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Sanity check
	//diagonalPreconditionerSanityCheck();
}

void surfelwarp::PreconditionerRhsBuilder::ComputeDiagonalPreconditionerGlobalIteration(cudaStream_t stream) {
	//Check the constants
	SURFELWARP_CHECK(m_penalty_constants.Density() < 1e-7f);

	//Do computation
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	m_block_preconditioner.ResizeArrayOrException(num_nodes * device::preconditioner_blk_size);
	dim3 blk(device::warp_size);
	dim3 grid(num_nodes);
	device::computeBlockDiagonalPreconditionerGlobalIterationKernel<<<grid, blk, 0, stream>>>(
		m_node2term_map,
		m_term2jacobian_map,
		m_block_preconditioner.Ptr(),
		m_penalty_constants
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	//Do inversion
	ComputeDiagonalPreconditionerInverse(stream);
}

void surfelwarp::PreconditionerRhsBuilder::ComputeDiagonalPreconditionerInverse(cudaStream_t stream) {
	m_preconditioner_inverse_handler->SetInput(m_block_preconditioner.ArrayView());
	m_preconditioner_inverse_handler->PerformDiagonalInverse(stream);
}