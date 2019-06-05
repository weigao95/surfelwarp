#include "common/ConfigParser.h"
#include "common/global_configs.h"
#include "common/Constants.h"
#include "common/device_intrinsics.h"
#include "core/warp_solver/solver_encode.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/JtJMaterializer.h"
#include "core/warp_solver/jtj_block_jacobian.cuh"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	enum {
		jtj_blk_size = 36,
		warp_size = 32,
	};

	

	__device__ __forceinline__ void fillScalarJtJToSharedBlock(
		const float jacobian[6],
		float shared_jtj_blks[jtj_blk_size][warp_size],
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
		float shared_jtj_blks[jtj_blk_size][warp_size],
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


	__device__ __forceinline__ void computeSmoothJtJBlock(
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned typed_term,
		unsigned encoded_pair,
		float shared_jtj_blks[jtj_blk_size][warp_size],
		float weight_square = 1.0f
	) {
		//Check the validity of this term
		const auto validity = term2jacobian.validity_indicator[typed_term];
		if(validity == 0) {
#pragma unroll
			for (int jac_row = 0; jac_row < 6; jac_row++) {
				shared_jtj_blks[6 * jac_row + 0][threadIdx.x] = 0.0f;
				shared_jtj_blks[6 * jac_row + 1][threadIdx.x] = 0.0f;
				shared_jtj_blks[6 * jac_row + 2][threadIdx.x] = 0.0f;
				shared_jtj_blks[6 * jac_row + 3][threadIdx.x] = 0.0f;
				shared_jtj_blks[6 * jac_row + 4][threadIdx.x] = 0.0f;
				shared_jtj_blks[6 * jac_row + 5][threadIdx.x] = 0.0f;
			}
			return;
		}

		//Explicit compute jacobian
		const float3 r = term2jacobian.Ti_xj[typed_term];
		const float3 s = term2jacobian.Tj_xj[typed_term];
		const ushort2 node_ij = term2jacobian.node_graph[typed_term];
		unsigned node_i, node_j;
		decode_nodepair(encoded_pair, node_i, node_j);
		
		//The order of two terms
		const float* jacobian_encoded_i;
		const float* jacobian_encoded_j;
		TwistGradientOfScalarCost twist_gradient_i, twist_gradient_j;
		if(node_i == node_ij.x) {
			jacobian_encoded_i = (const float*)(&twist_gradient_i);
			jacobian_encoded_j = (const float*)(&twist_gradient_j);
		} else {
			jacobian_encoded_i = (const float*)(&twist_gradient_j);
			jacobian_encoded_j = (const float*)(&twist_gradient_i);
		}

		//The first iteration assign
		{
			twist_gradient_i.rotation = make_float3(0.0f, r.z, -r.y);
			twist_gradient_i.translation = make_float3(1.0f, 0.0f, 0.0f);
			twist_gradient_j.rotation = make_float3(0.0f, -s.z, s.y);
			twist_gradient_j.translation = make_float3(-1.0f,  0.0f,  0.0f);
#pragma unroll
			for (int jac_row = 0; jac_row < 6; jac_row++) {
				shared_jtj_blks[6 * jac_row + 0][threadIdx.x] = weight_square * jacobian_encoded_i[0] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 1][threadIdx.x] = weight_square * jacobian_encoded_i[1] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 2][threadIdx.x] = weight_square * jacobian_encoded_i[2] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 3][threadIdx.x] = weight_square * jacobian_encoded_i[3] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 4][threadIdx.x] = weight_square * jacobian_encoded_i[4] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 5][threadIdx.x] = weight_square * jacobian_encoded_i[5] * jacobian_encoded_j[jac_row];
			}
		}

		//The next two iterations, plus
		{
			twist_gradient_i.rotation = make_float3(-r.z, 0.0f, r.x);
			twist_gradient_i.translation = make_float3(0.0f, 1.0f, 0.0f);
			twist_gradient_j.rotation = make_float3(s.z, 0.0f, -s.x);
			twist_gradient_j.translation = make_float3( 0.0f, -1.0f,  0.0f);
#pragma unroll
			for (int jac_row = 0; jac_row < 6; jac_row++) {
				shared_jtj_blks[6 * jac_row + 0][threadIdx.x] += weight_square * jacobian_encoded_i[0] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 1][threadIdx.x] += weight_square * jacobian_encoded_i[1] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 2][threadIdx.x] += weight_square * jacobian_encoded_i[2] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 3][threadIdx.x] += weight_square * jacobian_encoded_i[3] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 4][threadIdx.x] += weight_square * jacobian_encoded_i[4] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 5][threadIdx.x] += weight_square * jacobian_encoded_i[5] * jacobian_encoded_j[jac_row];
			}			
		}


		{
			twist_gradient_i.rotation = make_float3(r.y, -r.x, 0.0f);
			twist_gradient_i.translation = make_float3(0.0f, 0.0f, 1.0f);
			twist_gradient_j.rotation = make_float3(-s.y, s.x, 0.0f);
			twist_gradient_j.translation = make_float3(0.0f,  0.0f, -1.0f);
#pragma unroll
			for (int jac_row = 0; jac_row < 6; jac_row++) {
				shared_jtj_blks[6 * jac_row + 0][threadIdx.x] += weight_square * jacobian_encoded_i[0] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 1][threadIdx.x] += weight_square * jacobian_encoded_i[1] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 2][threadIdx.x] += weight_square * jacobian_encoded_i[2] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 3][threadIdx.x] += weight_square * jacobian_encoded_i[3] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 4][threadIdx.x] += weight_square * jacobian_encoded_i[4] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 5][threadIdx.x] += weight_square * jacobian_encoded_i[5] * jacobian_encoded_j[jac_row];
			}			
		}
	}

	__device__ __forceinline__ void fillChannelledJtJToSharedBlock(
		const float jacobian_channelled[18],
		float shared_jtj_blks[jtj_blk_size][warp_size],
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


	//The deprecated method
	__device__ __forceinline__ void computeSmoothJtJBlockOnline(
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned typed_term,
		unsigned encoded_pair,
		float shared_jtj_blks[jtj_blk_size][warp_size],
		float weight_square = 1.0f
	) {
		const ushort2 node_ij = term2jacobian.node_graph[typed_term];
		const auto xj4 = term2jacobian.reference_node_coords[node_ij.y];
		DualQuaternion dq_i = term2jacobian.node_se3[node_ij.x];
		DualQuaternion dq_j = term2jacobian.node_se3[node_ij.y];
		const mat34 Ti = dq_i.se3_matrix();
		const mat34 Tj = dq_j.se3_matrix();
		unsigned node_i, node_j;
		decode_nodepair(encoded_pair, node_i, node_j);

		//Explicit compute jacobian
		const float3 xj = make_float3(xj4.x, xj4.y, xj4.z);
		const float3 r = Ti.rot * xj + Ti.trans;
		const float3 s = Tj.rot * xj + Tj.trans;
		TwistGradientOfScalarCost twist_gradient_i, twist_gradient_j;
		
		//The order of two terms
		const float* jacobian_encoded_i;
		const float* jacobian_encoded_j;
		if(node_i == node_ij.x) {
			jacobian_encoded_i = (const float*)(&twist_gradient_i);
			jacobian_encoded_j = (const float*)(&twist_gradient_j);
		} else {
			jacobian_encoded_i = (const float*)(&twist_gradient_j);
			jacobian_encoded_j = (const float*)(&twist_gradient_i);
		}

		//The first iteration assign
		{
			twist_gradient_i.rotation = make_float3(0.0f, r.z, -r.y);
			twist_gradient_i.translation = make_float3(1.0f, 0.0f, 0.0f);
			twist_gradient_j.rotation = make_float3(0.0f, -s.z, s.y);
			twist_gradient_j.translation = make_float3(-1.0f,  0.0f,  0.0f);
#pragma unroll
			for (int jac_row = 0; jac_row < 6; jac_row++) {
				shared_jtj_blks[6 * jac_row + 0][threadIdx.x] = weight_square * jacobian_encoded_i[0] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 1][threadIdx.x] = weight_square * jacobian_encoded_i[1] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 2][threadIdx.x] = weight_square * jacobian_encoded_i[2] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 3][threadIdx.x] = weight_square * jacobian_encoded_i[3] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 4][threadIdx.x] = weight_square * jacobian_encoded_i[4] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 5][threadIdx.x] = weight_square * jacobian_encoded_i[5] * jacobian_encoded_j[jac_row];
			}
		}

		//The next two iterations, plus
		{
			twist_gradient_i.rotation = make_float3(-r.z, 0.0f, r.x);
			twist_gradient_i.translation = make_float3(0.0f, 1.0f, 0.0f);
			twist_gradient_j.rotation = make_float3(s.z, 0.0f, -s.x);
			twist_gradient_j.translation = make_float3( 0.0f, -1.0f,  0.0f);
#pragma unroll
			for (int jac_row = 0; jac_row < 6; jac_row++) {
				shared_jtj_blks[6 * jac_row + 0][threadIdx.x] += weight_square * jacobian_encoded_i[0] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 1][threadIdx.x] += weight_square * jacobian_encoded_i[1] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 2][threadIdx.x] += weight_square * jacobian_encoded_i[2] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 3][threadIdx.x] += weight_square * jacobian_encoded_i[3] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 4][threadIdx.x] += weight_square * jacobian_encoded_i[4] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 5][threadIdx.x] += weight_square * jacobian_encoded_i[5] * jacobian_encoded_j[jac_row];
			}			
		}


		{
			twist_gradient_i.rotation = make_float3(r.y, -r.x, 0.0f);
			twist_gradient_i.translation = make_float3(0.0f, 0.0f, 1.0f);
			twist_gradient_j.rotation = make_float3(-s.y, s.x, 0.0f);
			twist_gradient_j.translation = make_float3(0.0f,  0.0f, -1.0f);
#pragma unroll
			for (int jac_row = 0; jac_row < 6; jac_row++) {
				shared_jtj_blks[6 * jac_row + 0][threadIdx.x] += weight_square * jacobian_encoded_i[0] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 1][threadIdx.x] += weight_square * jacobian_encoded_i[1] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 2][threadIdx.x] += weight_square * jacobian_encoded_i[2] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 3][threadIdx.x] += weight_square * jacobian_encoded_i[3] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 4][threadIdx.x] += weight_square * jacobian_encoded_i[4] * jacobian_encoded_j[jac_row];
				shared_jtj_blks[6 * jac_row + 5][threadIdx.x] += weight_square * jacobian_encoded_i[5] * jacobian_encoded_j[jac_row];
			}			
		}
	}


	__global__ void computeJtJNonDiagonalBlockKernel(
		const NodePair2TermsIndex::NodePair2TermMap nodepair2term,
		const Term2JacobianMaps term2jacobian,
		float* jtj_blks,
		const PenaltyConstants constants = PenaltyConstants()
	) {
		const auto nodepair_idx = blockIdx.x;
		const auto encoded_pair = nodepair2term.encoded_nodepair[nodepair_idx];
		const auto term_begin = nodepair2term.nodepair_term_range[nodepair_idx].x;
		const auto term_end = nodepair2term.nodepair_term_range[nodepair_idx].y;
		const auto term_size = term_end - term_begin;
		const auto padded_term_size = warp_size * ((term_size + warp_size - 1) / warp_size);
		const auto lane_id = threadIdx.x & 31;

		//The memory for store the JtResidual result of each threads
		__shared__ float shared_blks[jtj_blk_size][warp_size];
		//The memory to perform the reduction
		__shared__ float reduced_blks[jtj_blk_size];

		//Zero out the elements
		for(auto iter = threadIdx.x; iter < jtj_blk_size; iter += warp_size)
			reduced_blks[iter] = 0.0f;
		__syncthreads();

		
		for (auto iter = threadIdx.x; iter < padded_term_size; iter += warp_size)
		{
			//The global term index
			bool term_valid = true;
			
			if(iter < term_size)
			{
				const auto term_idx = nodepair2term.nodepair_term_index[term_begin + iter];
				unsigned typed_term_idx;
				TermType term_type;
				query_typed_index(term_idx, nodepair2term.term_offset, term_type, typed_term_idx);

				switch (term_type) {
				case TermType::DenseImage:
					{
						float term_jacobian[6] = {0};
						float nodepair_weight = 0;
						computeScalarJtJBlockJacobian(term2jacobian.dense_depth_term, encoded_pair, typed_term_idx, term_jacobian, &nodepair_weight);
						fillScalarJtJToSharedBlock(term_jacobian, shared_blks, constants.DenseDepthSquared() * nodepair_weight);
#if defined(USE_DENSE_IMAGE_DENSITY_TERM)
						computeScalarJtJBlockJacobian(term2jacobian.density_map_term, encoded_pair, typed_term_idx, term_jacobian, &nodepair_weight);
						incrementScalarJtJToSharedBlock(term_jacobian, shared_blks, constants.DensitySquared() * nodepair_weight);
#endif
					}
					break;
				case TermType::Smooth:
					computeSmoothJtJBlock(term2jacobian.smooth_term, typed_term_idx, encoded_pair, shared_blks, constants.SmoothSquared());
					break;
				case TermType::Foreground:
					{
						float term_jacobian[6] = {0};
						float nodepair_weight = 0;
						computeScalarJtJBlockJacobian(term2jacobian.foreground_mask_term, encoded_pair, typed_term_idx, term_jacobian, &nodepair_weight);
						fillScalarJtJToSharedBlock(term_jacobian, shared_blks, constants.ForegroundSquared() * nodepair_weight);
					}
					break;
				case TermType::Feature:
					{
						float term_jacobian[18] = {0};
						float nodepair_weight = 0;
						computeFeatureJtJBlockJacobian(term2jacobian.sparse_feature_term, encoded_pair, typed_term_idx, term_jacobian, &nodepair_weight);
						fillChannelledJtJToSharedBlock(term_jacobian, shared_blks, constants.SparseFeatureSquared() * nodepair_weight);
					}
					break;
				default:
					term_valid = false;
				}
			}

			//__syncthreads();

			//Do a reduction
			for (int i = 0; i < jtj_blk_size; i++) {
				float data = (iter < term_size && term_valid) ? shared_blks[i][threadIdx.x] : 0.0f;
				data = warp_scan(data);
				if (lane_id == warpSize - 1) {
					reduced_blks[i] += data;
				}
			}
		}

		//Write to output
		for(auto iter = threadIdx.x; iter < jtj_blk_size; iter += warp_size)
			jtj_blks[jtj_blk_size * nodepair_idx + iter] = reduced_blks[iter];
	}



	__global__ void computeJtJNonDiagonalBlockGlobalIterationKernel(
		const NodePair2TermsIndex::NodePair2TermMap nodepair2term,
		const Term2JacobianMaps term2jacobian,
		float* jtj_blks,
		const PenaltyConstants constants = PenaltyConstants()
	) {
		const auto nodepair_idx = blockIdx.x;
		const auto encoded_pair = nodepair2term.encoded_nodepair[nodepair_idx];
		const auto term_begin = nodepair2term.nodepair_term_range[nodepair_idx].x;
		const auto term_end = nodepair2term.nodepair_term_range[nodepair_idx].y;
		const auto term_size = term_end - term_begin;
		const auto padded_term_size = warp_size * ((term_size + warp_size - 1) / warp_size);
		const auto lane_id = threadIdx.x & 31;

		//The memory for store the JtResidual result of each threads
		__shared__ float shared_blks[jtj_blk_size][warp_size];
		//The memory to perform the reduction
		__shared__ float reduced_blks[jtj_blk_size];

		//Zero out the elements
		for (auto iter = threadIdx.x; iter < jtj_blk_size; iter += warp_size)
			reduced_blks[iter] = 0.0f;
		__syncthreads();


		for (auto iter = threadIdx.x; iter < padded_term_size; iter += warp_size)
		{
			//The global term index
			bool term_valid = true;

			if (iter < term_size)
			{
				const auto term_idx = nodepair2term.nodepair_term_index[term_begin + iter];
				unsigned typed_term_idx;
				TermType term_type;
				query_typed_index(term_idx, nodepair2term.term_offset, term_type, typed_term_idx);

				switch (term_type) {
				case TermType::DenseImage:
				{
					float term_jacobian[6] = { 0 };
					float nodepair_weight = 0;
					computeScalarJtJBlockJacobian(term2jacobian.dense_depth_term, encoded_pair, typed_term_idx, term_jacobian, &nodepair_weight);
					fillScalarJtJToSharedBlock(term_jacobian, shared_blks, constants.DenseDepthSquared() * nodepair_weight);
				}
				break;
				case TermType::Smooth:
					computeSmoothJtJBlock(term2jacobian.smooth_term, typed_term_idx, encoded_pair, shared_blks, constants.SmoothSquared());
					break;
				case TermType::Foreground:
				{
					float term_jacobian[6] = { 0 };
					float nodepair_weight = 0;
					computeScalarJtJBlockJacobian(term2jacobian.foreground_mask_term, encoded_pair, typed_term_idx, term_jacobian, &nodepair_weight);
					fillScalarJtJToSharedBlock(term_jacobian, shared_blks, constants.ForegroundSquared() * nodepair_weight);
				}
				break;
				case TermType::Feature:
				{
					float term_jacobian[18] = { 0 };
					float nodepair_weight = 0;
					computeFeatureJtJBlockJacobian(term2jacobian.sparse_feature_term, encoded_pair, typed_term_idx, term_jacobian, &nodepair_weight);
					fillChannelledJtJToSharedBlock(term_jacobian, shared_blks, constants.SparseFeatureSquared() * nodepair_weight);
				}
				break;
				default:
					term_valid = false;
				}
			}

			//__syncthreads();

			//Do a reduction
			for (int i = 0; i < jtj_blk_size; i++) {
				float data = (iter < term_size && term_valid) ? shared_blks[i][threadIdx.x] : 0.0f;
				data = warp_scan(data);
				if (lane_id == warpSize - 1) {
					reduced_blks[i] += data;
				}
			}
		}

		//Write to output
		for (auto iter = threadIdx.x; iter < jtj_blk_size; iter += warp_size)
			jtj_blks[jtj_blk_size * nodepair_idx + iter] = reduced_blks[iter];
	}


	//For performance test
	__global__ void memoryPerformanceTestKernel(
		DeviceArraySlice<float> jtj_blks
	) {
		const auto nodepair_idx = blockIdx.x;
		//Write to output
		for(auto iter = threadIdx.x; iter < jtj_blk_size; iter += warp_size)
			jtj_blks[jtj_blk_size * nodepair_idx + iter] = nodepair_idx;
	}

} // namespace device
} // namespace surfelwarp




void surfelwarp::JtJMaterializer::computeNonDiagonalBlocks(cudaStream_t stream)
{
	//Correct the size of node pairs
	const auto num_nodepairs = m_nodepair2term_map.encoded_nodepair.Size();
	SURFELWARP_CHECK_EQ(num_nodepairs, m_nodepair2term_map.nodepair_term_range.Size());
	m_nondiag_blks.ResizeArrayOrException(num_nodepairs * device::jtj_blk_size);
	
	//Invoke the kernel
	dim3 blk(device::warp_size);
	dim3 grid(num_nodepairs);
	device::computeJtJNonDiagonalBlockKernel<<<grid, blk, 0, stream>>>(
		m_nodepair2term_map,
		m_term2jacobian_map,
		m_nondiag_blks.Ptr(),
		m_penalty_constants
	);

	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Do a sanity check
	//nonDiagonalBlocksSanityCheck();
}


void surfelwarp::JtJMaterializer::BuildMaterializedJtJNondiagonalBlocksGlobalIteration(cudaStream_t stream) {
	//Correct the size of node pairs
	const auto num_nodepairs = m_nodepair2term_map.encoded_nodepair.Size();
	SURFELWARP_CHECK_EQ(num_nodepairs, m_nodepair2term_map.nodepair_term_range.Size());
	m_nondiag_blks.ResizeArrayOrException(num_nodepairs * device::jtj_blk_size);

	//Invoke the kernel
	dim3 blk(device::warp_size);
	dim3 grid(num_nodepairs);
	device::computeJtJNonDiagonalBlockGlobalIterationKernel<<<grid, blk, 0, stream>>>(
		m_nodepair2term_map,
		m_term2jacobian_map,
		m_nondiag_blks.Ptr(),
		m_penalty_constants
	);


	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}