#include "common/global_configs.h"
#include "common/sanity_check.h"
#include "core/warp_solver/term_offset_types.h"
#include "core/warp_solver/solver_encode.h"
#include "core/warp_solver/NodePair2TermsIndex.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	__host__ __device__ __forceinline__ 
	void computeKVNodePairKNN(
		const ushort4& knn,
		unsigned* nodepair_key
	) {
		const unsigned short* knn_arr = (const unsigned short*)(&knn);
		auto offset = 0;
		for(auto i = 0; i < 4; i++) {
			const auto node_i = knn_arr[i];
			for(auto j = 0; j < 4; j++) {
				const auto node_j = knn_arr[j];
				if(node_i < node_j) {
					nodepair_key[offset] = encode_nodepair(node_i, node_j);
					offset++;
				}
			}
		}
	}
	
	__global__ void buildKeyValuePairKernel(
		DeviceArrayView<ushort4> dense_image_knn,
		DeviceArrayView<ushort2> node_graph,
		//These terms might be empty
		DeviceArrayView<ushort4> foreground_mask_knn,
		DeviceArrayView<ushort4> sparse_feature_knn,
		const TermTypeOffset offset,
		//The output
		unsigned* nodepair_keys,
		unsigned* term_values
	) {
		const auto term_idx = threadIdx.x + blockIdx.x * blockDim.x;
		TermType term_type;
		unsigned typed_term_idx, kv_offset;
		query_nodepair_index(term_idx, offset, term_type, typed_term_idx, kv_offset);
		
		//Compute the pair key locally
		unsigned term_nodepair_key[6];
		unsigned save_size = 6;
		
		//Zero init
#pragma unroll
		for(auto i = 0; i < 6; i++) {
			term_nodepair_key[i] = 0xFFFFFFFF;
		}

		switch (term_type) {
		case TermType::DenseImage:
			computeKVNodePairKNN(dense_image_knn[typed_term_idx], term_nodepair_key);
			break;
		case TermType::Smooth:
			{
				const auto node_pair = node_graph[typed_term_idx];
				if(node_pair.x < node_pair.y) {
					term_nodepair_key[0] = encode_nodepair(node_pair.x, node_pair.y);
				} else if(node_pair.y < node_pair.x) {
					term_nodepair_key[0] = encode_nodepair(node_pair.y, node_pair.x);
				}
				save_size = 1;
			}
			break;
		case TermType::Foreground:
			computeKVNodePairKNN(foreground_mask_knn[typed_term_idx], term_nodepair_key);
			break;
		case TermType::Feature:
			computeKVNodePairKNN(sparse_feature_knn[typed_term_idx], term_nodepair_key);
			break;
		default:
			save_size = 0;
			break;
		}

		//Save it
		for(auto i = 0; i < save_size; i++) {
			nodepair_keys[kv_offset + i] = term_nodepair_key[i];
			term_values[kv_offset + i] = term_idx;
		}
	}


	__global__ void segmentNodePairKernel(
		const DeviceArrayView<unsigned> sorted_node_pair,
		unsigned* segment_label
	) {
		//Check the valid of node size
		const auto idx = threadIdx.x + blockDim.x*blockIdx.x;
		if(idx >= sorted_node_pair.Size()) return;

		//The label must be written
		unsigned label = 0;

		//Check the size of node pair
		const auto encoded_pair = sorted_node_pair[idx];
		unsigned node_i, node_j;
		decode_nodepair(encoded_pair, node_i, node_j);
		if(encoded_pair == 0xFFFFFFFF || node_i > 4096 || node_j > 4096) {
			//pass
		} else {
			if (idx == 0) label = 1;
			else //Can check the prev one
			{
				const auto encoded_prev = sorted_node_pair[idx - 1];
				if(encoded_prev != encoded_pair) label = 1;
			}
		}

		//Write to result
		segment_label[idx] = label;
	}


	__global__ void compactNodePairKeyKernel(
		const DeviceArrayView<unsigned> sorted_node_pair,
		const unsigned* segment_label,
		const unsigned* inclusive_sum_label,
		unsigned* compacted_key,
		unsigned* compacted_offset
	) {
		const auto idx = threadIdx.x + blockDim.x*blockIdx.x;
		if (idx < sorted_node_pair.Size() - 1) {
			if (segment_label[idx] > 0) {
				const auto compacted_idx = inclusive_sum_label[idx] - 1;
				compacted_key[compacted_idx] = sorted_node_pair[idx];
				compacted_offset[compacted_idx] = idx;
			}
		}
		else if (idx == sorted_node_pair.Size() - 1) {
			//The size of the sorted_key, segment label and 
			//inclusive-sumed segment are the same
			const auto last_idx = inclusive_sum_label[idx];
			compacted_offset[last_idx] = sorted_node_pair.Size();
			if (segment_label[idx] > 0) {
				const auto compacted_idx = last_idx - 1;
				compacted_key[compacted_idx] = sorted_node_pair[idx];
				compacted_offset[compacted_idx] = idx;
			}
		}
	}

	__global__ void computeSymmetricNodePairKernel(
		const DeviceArrayView<unsigned> compacted_key,
		const unsigned* compacted_offset,
		unsigned* full_nodepair_key,
		uint2* full_term_start_end
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < compacted_key.Size()) {
			const unsigned nodepair = compacted_key[idx];
			const unsigned start_idx = compacted_offset[idx];
			const unsigned end_idx = compacted_offset[idx + 1]; //This is safe
			unsigned node_i, node_j;
			decode_nodepair(nodepair, node_i, node_j);
			const unsigned sym_nodepair = encode_nodepair(node_j, node_i);
			full_nodepair_key[2 * idx + 0] = nodepair;
			full_nodepair_key[2 * idx + 1] = sym_nodepair;
			full_term_start_end[2 * idx + 0] = make_uint2(start_idx, end_idx);
			full_term_start_end[2 * idx + 1] = make_uint2(start_idx, end_idx);
		}
	}

} // namespace device
} // namespace surfelwarp

void surfelwarp::NodePair2TermsIndex::buildTermKeyValue(cudaStream_t stream) {
	//Correct the size of array
	const auto num_kvs = NumKeyValuePairs();
	m_nodepair_keys.ResizeArrayOrException(num_kvs);
	m_term_idx_values.ResizeArrayOrException(num_kvs);
	
	const auto num_terms = NumTerms();
	dim3 blk(256);
	dim3 grid(divUp(num_terms, blk.x));
	device::buildKeyValuePairKernel<<<grid, blk, 0, stream>>>(
		m_term2node.dense_image_knn,
		m_term2node.node_graph,
		m_term2node.foreground_mask_knn,
		m_term2node.sparse_feature_knn,
		m_term_offset,
		m_nodepair_keys.Ptr(),
		m_term_idx_values.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::NodePair2TermsIndex::sortCompactTermIndex(cudaStream_t stream) {
	m_nodepair2term_sorter.Sort(m_nodepair_keys.ArrayView(), m_term_idx_values.ArrayView(), 24, stream);
	
	//Do segmentation
	m_segment_label.ResizeArrayOrException(m_nodepair_keys.ArraySize());
	DeviceArrayView<unsigned> sorted_node_pair(m_nodepair2term_sorter.valid_sorted_key);
	dim3 blk(256);
	dim3 grid(divUp(sorted_node_pair.Size(), blk.x));
	device::segmentNodePairKernel<<<grid, blk, 0, stream>>>(sorted_node_pair, m_segment_label.Ptr());

	//Do prefix sum and compaction
	m_segment_label_prefixsum.InclusiveSum(m_segment_label.ArrayView(), stream);
	device::compactNodePairKeyKernel<<<grid, blk, 0, stream>>>(
		sorted_node_pair,
		m_segment_label.Ptr(),
		m_segment_label_prefixsum.valid_prefixsum_array.ptr(),
		m_half_nodepair_keys.Ptr(),
		m_half_nodepair2term_offset.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Debug
	//std::vector<unsigned> sorted_key;
	//m_nodepair_keys.ArrayView().Download(sorted_key);
	//std::cout << sorted_key[sorted_key.size() - 1] << std::endl;
}

void surfelwarp::NodePair2TermsIndex::buildSymmetricCompactedIndex(cudaStream_t stream) {
	//Assume the size has been queried
	dim3 blk(128);
	dim3 grid(divUp(m_half_nodepair_keys.ArraySize(), blk.x));
	device::computeSymmetricNodePairKernel<<<grid, blk, 0, stream>>>(
		m_half_nodepair_keys.ArrayView(),
		m_half_nodepair2term_offset.Ptr(),
		m_compacted_nodepair_keys.Ptr(),
		m_nodepair_term_range.Ptr()
	);
	
	//Sort the key-value pair
	m_symmetric_kv_sorter.Sort(m_compacted_nodepair_keys.ArrayView(), m_nodepair_term_range.ArrayView(), 24, stream);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::NodePair2TermsIndex::QueryValidNodePairSize(cudaStream_t stream) {
	const unsigned* num_unique_pair_dev = m_segment_label_prefixsum.valid_prefixsum_array.ptr() + (m_segment_label_prefixsum.valid_prefixsum_array.size() - 1);
	unsigned num_unique_pair;
	cudaSafeCall(cudaMemcpyAsync(&num_unique_pair, num_unique_pair_dev, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
	cudaSafeCall(cudaStreamSynchronize(stream));
	
	//Correct the size
	m_half_nodepair_keys.ResizeArrayOrException(num_unique_pair);
	m_half_nodepair2term_offset.ResizeArrayOrException(num_unique_pair + 1);
	m_compacted_nodepair_keys.ResizeArrayOrException(2 * num_unique_pair);
	m_nodepair_term_range.ResizeArrayOrException(2 * num_unique_pair);
}