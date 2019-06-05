#include "common/global_configs.h"
#include "common/ConfigParser.h"
#include "common/Constants.h"
#include "core/warp_solver/PenaltyConstants.h"
#include "core/warp_solver/ResidualEvaluator.h"
#include "core/warp_solver/term_offset_types.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include <cub/cub.cuh>
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	__device__ __forceinline__ void computeSmoothResidual(
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned typed_term,
		float residual[3]
	) {
		const auto Ti_xj = term2jacobian.Ti_xj[typed_term];
		const auto Tj_xj = term2jacobian.Tj_xj[typed_term];
		const auto validity = term2jacobian.validity_indicator[typed_term];
		if(validity == 0) {
			residual[0] = residual[1] = residual[2] = 0.0f;
			return;
		}
		computeSmoothTermResidual(Ti_xj, Tj_xj, residual);
	}


	__device__ __forceinline__ void computeSmoothResidualOnline(
		const NodeGraphSmoothTerm2Jacobian& term2jacobian,
		unsigned typed_term,
		float residual[3]
	) {
		const ushort2 node_ij = term2jacobian.node_graph[typed_term];
		const auto xi = term2jacobian.reference_node_coords[node_ij.x];
		const auto xj = term2jacobian.reference_node_coords[node_ij.y];
		DualQuaternion dq_i = term2jacobian.node_se3[node_ij.x];
		DualQuaternion dq_j = term2jacobian.node_se3[node_ij.y];
		const mat34 Ti = dq_i.se3_matrix();
		const mat34 Tj = dq_j.se3_matrix();
		const auto validity = term2jacobian.validity_indicator[typed_term];
		if(validity == 0) {
			residual[0] = residual[1] = residual[2] = 0.0f;
			return;
		}
		computeSmoothTermResidual(xj, Ti, Tj, residual);
	}

	__device__ __forceinline__ void computeFeatureResidual(
		const Point2PointICPTerm2Jacobian& term2jacobian,
		unsigned typed_term_idx,
		float residual[3]
	) {
		const float4 target_vertex = term2jacobian.target_vertex[typed_term_idx];
		const float4 warped_vertex = term2jacobian.warped_vertex[typed_term_idx];
		computePointToPointICPTermResidual(target_vertex, warped_vertex, residual);
	}
	
	__global__ void computeTermResidualKernel(
		const TermTypeOffset term_offset,
		const Term2JacobianMaps term2jacobian,
		//The output
		float* residuals,
		bool use_square,
		const PenaltyConstants constants = PenaltyConstants()
	) {
		//Parallel over all terms
		const auto term_idx = threadIdx.x + blockDim.x * blockIdx.x;
		
		//Query the term type
		unsigned typed_term_idx, scalar_term_idx;
		TermType term_type;
		query_typed_index(term_idx, term_offset, term_type, typed_term_idx, scalar_term_idx);

		//Depending on the types
		switch (term_type)
		{
		case TermType::DenseImage: {
				auto value = constants.DenseDepth() * term2jacobian.dense_depth_term.residual_array[typed_term_idx];
				value = fabsf(value);
				if(use_square) value = value * value;
#if defined(USE_DENSE_IMAGE_DENSITY_TERM)
				auto density_value = constants.Density() * term2jacobian.density_map_term.residual_array[typed_term_idx];
				density_value = fabsf(density_value);
				if (use_square) density_value = density_value * density_value;
				value += density_value;
#endif
				residuals[scalar_term_idx] = value;
			}
			break;
		case TermType::Smooth: {
				float smooth_residual[3];
				computeSmoothResidual(term2jacobian.smooth_term, typed_term_idx, smooth_residual);
				for(auto i = 0; i < 3; i++) {
					auto value = constants.Smooth() * smooth_residual[i];
					if(use_square) value = value * value;
					residuals[scalar_term_idx + i] = value;
				}
			}
			break;
		/*case TermType::DensityMap: {
				auto value = constants.Density() * term2jacobian.density_map_term.residual_array[typed_term_idx];
				if(use_square) value = value * value;
				residuals[scalar_term_idx] = value;
			}
			break;*/
		case TermType::Foreground: {
				auto value = constants.Foreground() * term2jacobian.foreground_mask_term.residual_array[typed_term_idx];
				if(use_square) value = value * value;
				residuals[scalar_term_idx] = value;
			}
			break;
		case TermType::Feature: {
				float feature_residual[3];
				computeFeatureResidual(term2jacobian.sparse_feature_term, typed_term_idx, feature_residual);
				for(auto i = 0; i < 3; i++) {
					auto value = constants.SparseFeature() * feature_residual[i];
					if(use_square) value = value * value;
					residuals[scalar_term_idx + i] = value;
				}
			}
			break;
		default:
			break;
		}
	}

} // namespace device
} // namespace surfelwarp

//The create required access to cuda cc
void surfelwarp::ResidualEvaluator::AllocateBuffer() {
	const auto& config = ConfigParser::Instance();
	const auto num_pixels = config.clip_image_cols() * config.clip_image_rows();
	const auto max_dense_depth_terms = num_pixels;
	const auto max_node_graph_terms = Constants::kMaxNumNodes * Constants::kNumNodeGraphNeigbours;
	const auto max_density_terms = num_pixels;
	const auto max_foreground_terms = num_pixels / 2; //Only part on boundary
	const auto max_feature_terms = Constants::kMaxMatchedSparseFeature;
	
	//The residual by terms
	const auto scalar_term_size = max_dense_depth_terms + max_density_terms + max_foreground_terms + 3 * (max_node_graph_terms + max_feature_terms);
	m_termwise_residual.AllocateBuffer(scalar_term_size);
	
	//The residual by nodes
	m_nodewise_residual.AllocateBuffer(Constants::kMaxNumNodes);
	
	//The prefix sum buffer
	m_residual_prefixsum.AllocateBuffer(scalar_term_size);
	
	//Query the buffer size required by cub
	size_t prefixsum_bytes = 0;
	cub::DeviceScan::InclusiveSum(
		NULL, prefixsum_bytes,
		m_termwise_residual.Ptr(), m_residual_prefixsum.Ptr(),
		(int) scalar_term_size, 0
	);
	m_prefixsum_buffer.create(prefixsum_bytes);
	
	//The page-locked host memory
	cudaSafeCall(cudaMallocHost((void**)(&m_residual_value_pagelock), sizeof(float)));
}

void surfelwarp::ResidualEvaluator::ReleaseBuffer() {
	m_termwise_residual.ReleaseBuffer();
	m_nodewise_residual.ReleaseBuffer();
	m_residual_prefixsum.ReleaseBuffer();
	m_prefixsum_buffer.release();
	cudaSafeCall(cudaFreeHost(m_residual_value_pagelock));
}


void surfelwarp::ResidualEvaluator::ComputeResidualByTerms(cudaStream_t stream) {
	const auto scalar_term_size = m_node2term_map.term_offset.ScalarTermSize();
	m_termwise_residual.ResizeArrayOrException(scalar_term_size);
	const auto use_square = true;
	
	dim3 blk(128);
	dim3 grid(divUp(scalar_term_size, blk.x));
	device::computeTermResidualKernel<<<grid, blk, 0, stream>>>(
		m_node2term_map.term_offset,
		m_term2jacobian_map,
		m_termwise_residual.Ptr(),
		use_square,
		m_penalty_constants
	);
}

void surfelwarp::ResidualEvaluator::CollectResidualByNodes(cudaStream_t stream) {

}

void surfelwarp::ResidualEvaluator::CollectTotalResidual(cudaStream_t stream) {
	m_residual_prefixsum.ResizeArrayOrException(m_termwise_residual.ArraySize());
	
	//Do perfix sum
	size_t prefixsum_bytes = m_prefixsum_buffer.sizeBytes();
	cub::DeviceScan::InclusiveSum(
		m_prefixsum_buffer.ptr(), prefixsum_bytes,
		m_termwise_residual.Ptr(), m_residual_prefixsum.Ptr(), m_termwise_residual.ArraySize(),
		stream
	);
}

float surfelwarp::ResidualEvaluator::SyncQueryTotalResidualHost(cudaStream_t stream) {
	const float* dev_ptr = m_residual_prefixsum.Ptr() + m_residual_prefixsum.ArraySize() - 1;
	cudaSafeCall(cudaMemcpyAsync(m_residual_value_pagelock, dev_ptr, sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaSafeCall(cudaStreamSynchronize(stream));
	return *m_residual_value_pagelock;
}
