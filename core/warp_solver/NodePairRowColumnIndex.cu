#include "common/device_intrinsics.h"
#include "pcg_solver/solver_configs.h"
#include "core/warp_solver/term_offset_types.h"
#include "core/warp_solver/solver_encode.h"
#include "core/warp_solver/NodePair2TermsIndex.h"
#include <device_launch_parameters.h>
#include <math_functions.h>

namespace surfelwarp { namespace device {

	//Kernel for computing of the row offset in node_pair array
	__global__ void computeRowOffsetKernel(
		const DeviceArrayView<unsigned> compacted_Iij_key,
		DeviceArraySlice<unsigned> rowoffset_array
	) {
		const auto idx = threadIdx.x + blockIdx.x*blockDim.x;
		if (idx >= compacted_Iij_key.Size()) return;
		if (idx == 0) {
			rowoffset_array[0] = 0;
			rowoffset_array[rowoffset_array.Size() - 1] = compacted_Iij_key.Size();
		}
		else {
			const auto key_prev = compacted_Iij_key[idx - 1];
			const auto key_this = compacted_Iij_key[idx];
			const auto row_prev = encoded_row(key_prev);
			const auto row_this = encoded_row(key_this);
			if (row_this != row_prev) {
				rowoffset_array[row_this] = idx;
			}
		}
	}

	//Kernel for computing the length of each row 
	// (both diag and non-diagonal terms)
	__global__ void computeRowBlockLengthKernel(
		const unsigned* rowoffset_array,
		DeviceArraySlice<unsigned> blk_rowlength
	) {
		const auto idx = threadIdx.x + blockDim.x*blockIdx.x;
		if (idx >= blk_rowlength.Size()) return;
		//Note that the diagonal term is included
		blk_rowlength[idx] = 1 + rowoffset_array[idx + 1] - rowoffset_array[idx];
	}

	__global__ void computeBinLengthKernel(
		const DeviceArrayView<unsigned> rowblk_length,
		DeviceArraySlice<unsigned> valid_bin_length,
		unsigned* valid_nonzeros_rowscan
	) {
		__shared__ unsigned partial_sum[32];

		//The idx is in [0, 1024)
		const auto idx = threadIdx.x + blockDim.x*blockIdx.x;
		const auto warp_idx = idx >> 5;
		const auto lane_idx = idx & 31;
		
		unsigned bin_length = 0;
		if (idx < valid_bin_length.Size()) {
			//32 * idx is the real-matrix begin row
			//so does 32 * idx + 31 is the ending row
			//For a matrix row, its corresponding blk-row is
			//matrix_row / 6
			const unsigned blkrow_begin = bin_size * idx / 6;
			unsigned blkrow_end = (bin_size * idx + bin_size - 1) / 6;
			blkrow_end = umin(blkrow_end, rowblk_length.Size() - 1);
			unsigned max_length = 0;
			for (unsigned blkrow_idx = blkrow_begin; blkrow_idx <= blkrow_end; blkrow_idx++) {
				max_length = umax(max_length, rowblk_length[blkrow_idx]);
			}
			
			//From block length to actual element length
			bin_length = 6 * max_length;
			valid_bin_length[idx] = bin_length;
		}

		bin_length = warp_scan(bin_length);
		if (lane_idx == 31) {
			partial_sum[warp_idx] = bin_length;
		}
		__syncthreads();

		if (warp_idx == 0) {
			const unsigned partial_scan = partial_sum[lane_idx];
			partial_sum[lane_idx] = warp_scan(partial_scan);
		}
		__syncthreads();

		if (idx < valid_bin_length.Size()) {
			const unsigned offset = (warp_idx == 0 ? 0 : partial_sum[warp_idx - 1]);
			valid_nonzeros_rowscan[idx + 1] = 32 * (bin_length + offset);
		}
		
		//The first elements
		if (idx == 0) valid_nonzeros_rowscan[0] = 0;
	}


	__global__ void computeBinBlockedCSRRowPtrKernel(
		const unsigned* valid_nonzeros_rowscan,
		DeviceArraySlice<int> csr_rowptr
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= csr_rowptr.Size()) return;
		const int bin_row_idx = idx / 32;
		const int bin_row_offset = idx % 32;
		csr_rowptr[idx] = bin_row_offset + valid_nonzeros_rowscan[bin_row_idx];
	}


	//The column index for bin-block csr format
	__global__ void computeBinBlockedCSRColPtrKernel(
		const unsigned matrix_size,
		const int* csr_rowptr,
		const unsigned* compacted_nodepair,
		const unsigned* blkrow_offset,
		int* csr_colptr
	) {
		const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(row_idx >= matrix_size) return;

		//From now, the query on rowptr should be safe
		const auto blkrow_idx = row_idx / 6;
		const auto data_offset = csr_rowptr[row_idx];
		const auto lane_idx = threadIdx.x & 31;

		//For the diagonal terms
		auto column_idx_offset = (data_offset - lane_idx) / 6 + lane_idx;
		csr_colptr[column_idx_offset] = 6 * blkrow_idx;
		column_idx_offset += bin_size;

		//For the non-diagonal terms
		auto Iij_begin = blkrow_offset[blkrow_idx];
		const auto Iij_end = blkrow_offset[blkrow_idx + 1];

		for (; Iij_begin < Iij_end; Iij_begin++, column_idx_offset += bin_size) {
			const auto Iij_key = compacted_nodepair[Iij_begin];
			const auto blkcol_idx = encoded_col(Iij_key);
			csr_colptr[column_idx_offset] = 6 * blkcol_idx;
		}
	}

} // namespace device
} // namespace surfelwarp


void surfelwarp::NodePair2TermsIndex::computeBlockRowLength(cudaStream_t stream) {
	m_blkrow_offset_array.ResizeArrayOrException(m_num_nodes + 1);
	
	//Prepare the input
	DeviceArrayView<unsigned> compacted_nodepair(m_symmetric_kv_sorter.valid_sorted_key);
	dim3 offset_blk(128);
	dim3 offset_grid(divUp(compacted_nodepair.Size(), offset_blk.x));
	device::computeRowOffsetKernel<<<offset_grid, offset_blk, 0, stream>>>(
		compacted_nodepair,
		m_blkrow_offset_array.ArraySlice()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Compute the row length
	m_blkrow_length_array.ResizeArrayOrException(m_num_nodes);
	dim3 length_blk(64);
	dim3 length_grid(divUp(m_num_nodes, length_blk.x));
	device::computeRowBlockLengthKernel<<<length_grid, length_blk, 0, stream>>>(
		m_blkrow_offset_array.ArrayView(),
		m_blkrow_length_array.ArraySlice()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Debug sanity check
	//blockRowOffsetSanityCheck();
	//blockRowLengthSanityCheck();
}


void surfelwarp::NodePair2TermsIndex::computeBinLength(cudaStream_t stream) {
	//Correct the size of the matrix
	const auto matrix_size = m_num_nodes * 6;
	const auto num_bins = divUp(matrix_size, bin_size);
	
	m_binlength_array.ResizeArrayOrException(num_bins);
	m_binnonzeros_prefixsum.ResizeArrayOrException(num_bins + 1);
	device::computeBinLengthKernel<<<1, 1024, 0, stream>>>(
		m_blkrow_length_array.ArrayView(),
		m_binlength_array.ArraySlice(),
		m_binnonzeros_prefixsum.ArraySlice()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//The sanity check method
	//binLengthNonzerosSanityCheck();
}


void surfelwarp::NodePair2TermsIndex::computeBinBlockCSRRowPtr(cudaStream_t stream) {
	//Compute the row pointer in bin-blocked csr format
	m_binblocked_csr_rowptr.ResizeArrayOrException(32 * m_binnonzeros_prefixsum.ArraySize());
	dim3 rowptr_blk(128);
	dim3 rowptr_grid(divUp(m_binblocked_csr_rowptr.ArraySize(), rowptr_blk.x));
	device::computeBinBlockedCSRRowPtrKernel<<<rowptr_grid, rowptr_blk, 0, stream>>>(
		m_binnonzeros_prefixsum.Ptr(),
		m_binblocked_csr_rowptr.ArraySlice()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Sanity check method
	//binBlockCSRRowPtrSanityCheck();
}

void surfelwarp::NodePair2TermsIndex::nullifyBinBlockCSRColumePtr(cudaStream_t stream) {
	//Compute the size to nullify
	const auto total_blk_size = m_symmetric_kv_sorter.valid_sorted_key.size() + m_num_nodes;
	const auto nullify_size = std::min(7 * total_blk_size, m_binblocked_csr_colptr.BufferSize());
	
	//Do it
	cudaSafeCall(cudaMemsetAsync(
		m_binblocked_csr_colptr.Ptr(),
		0xFF,
		sizeof(int) * m_binblocked_csr_colptr.BufferSize(),
		stream
	));
}


void surfelwarp::NodePair2TermsIndex::computeBinBlockCSRColumnPtr(cudaStream_t stream) {
	//The compacted full nodepair array
	DeviceArrayView<unsigned> compacted_nodepair(m_symmetric_kv_sorter.valid_sorted_key);
	const auto matrix_size = 6 * m_num_nodes;
	
	//Do not need to query the size of colptr?
	dim3 colptr_blk(128);
	dim3 colptr_grid(divUp(32 * m_binlength_array.ArraySize(), colptr_blk.x));
	device::computeBinBlockedCSRColPtrKernel<<<colptr_grid, colptr_blk, 0, stream>>>(
		matrix_size,
		m_binblocked_csr_rowptr.Ptr(),
		compacted_nodepair.RawPtr(),
		m_blkrow_offset_array.Ptr(),
		m_binblocked_csr_colptr.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Debug method
	//binBlockCSRColumnPtrSanityCheck();
}