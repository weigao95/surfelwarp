#include "pcg_solver/solver_configs.h"
#include "core/warp_solver/JtJMaterializer.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	__global__ void assembleBinBlockCSRKernel(
		const unsigned matrix_size,
		const float* diagonal_blks,
		const float* nondiagonal_blks,
		const int* csr_rowptr,
		const unsigned* blkrow_offset,
		float* JtJ_data
	) {
		const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(row_idx >= matrix_size) return;

		//Now the query should all be safe
		int data_offset = csr_rowptr[row_idx];
		const auto blkrow_idx = row_idx / 6;
		const auto inblk_offset = row_idx % 6;

		//First fill the diagonal blks
		for (auto k = 0; k < 6; k++, data_offset += bin_size) {
			JtJ_data[data_offset] = diagonal_blks[36 * blkrow_idx + inblk_offset + 6 * k];
		}

		//Next fill the non-diagonal blks
		auto Iij_begin = blkrow_offset[blkrow_idx];
		const auto Iij_end = blkrow_offset[blkrow_idx + 1];
		for (; Iij_begin < Iij_end; Iij_begin++) {
			for (int k = 0; k < 6; k++, data_offset += bin_size) {
				JtJ_data[data_offset] = nondiagonal_blks[36 * Iij_begin + inblk_offset + 6 * k];
			}
		}
	}


} // device
} // surfelwarp

void surfelwarp::JtJMaterializer::AssembleBinBlockCSR(surfelwarp::DeviceArrayView<float> diagonal_blks, cudaStream_t stream) {
	//Zero out the matrix
	cudaSafeCall(cudaMemsetAsync(m_binblock_csr_data.Ptr(), 0, sizeof(float) * m_binblock_csr_data.Capacity(), stream));
	
	//The size of the matrix
	SURFELWARP_CHECK(diagonal_blks.Size() % 6 == 0);
	const auto matrix_size = diagonal_blks.Size() / 6;
	
	dim3 assemble_blk(128);
	dim3 assemble_grid(divUp(matrix_size, assemble_blk.x));
	device::assembleBinBlockCSRKernel<<<assemble_grid, assemble_blk, 0, stream>>>(
		matrix_size,
		diagonal_blks.RawPtr(),
		m_nondiag_blks.Ptr(),
		m_nodepair2term_map.binblock_csr_rowptr.RawPtr(),
		m_nodepair2term_map.blkrow_offset.RawPtr(),
		m_binblock_csr_data.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Construct the spmv applier
	m_spmv_handler->SetInputs(
		m_binblock_csr_data.Ptr(),
		m_nodepair2term_map.binblock_csr_rowptr.RawPtr(),
		m_nodepair2term_map.binblock_csr_colptr,
		matrix_size
	);
}