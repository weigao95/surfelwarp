#pragma once
#include "math/DenseGaussian.h"
#include "pcg_solver/BlockDiagonalPreconditionerInverse.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	template<int BlockDim, int NumThreads = 64>
	__global__ void blockDiagonalInverseKernel(
		const float* A,
		float* A_inversed,
		const unsigned num_matrix
	) {
		//Load the matrix into the shared memory
		const auto blk_size = BlockDim * BlockDim;
		__shared__ float factored_matrix[blk_size * NumThreads];
		__shared__ float inversed_matrix[blk_size * NumThreads];

		//The input matrix pointer for this block
		const int blk_matrix_offset = blk_size * blockDim.x * blockIdx.x;
		const float* A_this_blk = A + blk_matrix_offset;

		//Cooperative loading
		for (auto k = 0; k < blk_size; k++) {
			if (blk_matrix_offset + k * NumThreads + threadIdx.x < num_matrix * blk_size)
				factored_matrix[k * NumThreads + threadIdx.x] = A_this_blk[k * NumThreads + threadIdx.x]; //Each thread loads one element
		}

		//Sync here
		__syncthreads();

		//Call the Gaussian inversion
		float* A_this_thread = &(factored_matrix[blk_size * threadIdx.x]);
		float* A_inv_this_thread = &(inversed_matrix[blk_size * threadIdx.x]);
		DenseGaussian<BlockDim>::Inverse(A_this_thread, A_inv_this_thread);

		//Sync again
		__syncthreads();

		//Cooperative storing
		float* A_inv_this_blk = A_inversed + blk_matrix_offset;
		for (auto k = 0; k < blk_size; k++) {
			if (blk_matrix_offset + k * NumThreads + threadIdx.x < num_matrix * blk_size)
				A_inv_this_blk[k * NumThreads + threadIdx.x] = inversed_matrix[k * NumThreads + threadIdx.x]; //Each thread stores one element
		}
	}

} // namespace device
} // namespace surfelwarp

template<int BlockDim>
void surfelwarp::BlockDiagonalPreconditionerInverse<BlockDim>::PerformDiagonalInverse(cudaStream_t stream)
{
	const auto num_blks = m_matrix_size / BlockDim;
	dim3 inv_blk(64);
	dim3 inv_grid(divUp(num_blks, inv_blk.x));
	device::blockDiagonalInverseKernel<BlockDim, 64><<<inv_grid, inv_blk, 0, stream>>>(
		m_diagonal_blks.RawPtr(), m_inv_diag_blks.Ptr(), num_blks
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}
