#pragma once

#include "common/common_texture_utils.h"
#include "pcg_solver/BinBlockCSR.h"
#include "pcg_solver/ApplySpMVBinBlockCSR.h"
#include <device_launch_parameters.h>
#include <iostream>

namespace surfelwarp { namespace device {
	
	//The interface for spmv
	template<int BlockDim>
	__global__ void performSparseMVKernel(
		const float* A_data,
		const int* A_rowptr,
		const int* A_colptr,
		const float* d,
		DeviceArraySlice<float> q
	) {
		const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(row_idx < q.Size()) {
			//Compute SparseMV
			const auto spmv = BinBlockCSR<BlockDim>::SparseMV(A_data, A_colptr, A_rowptr, d, row_idx);
			q[row_idx] = spmv;
		}
	}

	template<int BlockDim>
	__global__ void performSparseMVKernel(
		const float* A_data,
		const int* A_rowptr,
		const int* A_colptr,
		cudaTextureObject_t d,
		DeviceArraySlice<float> q
	) {
		const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (row_idx < q.Size()) {
			//Compute SparseMV
			const auto spmv = BinBlockCSR<BlockDim>::SparseMV(A_data, A_colptr, A_rowptr, d, row_idx);
			q[row_idx] = spmv;
		}
	}



	//The interface for init residual
	template<int BlockDim>
	__global__ void initializeResidualKernel(
		const DeviceArrayView<float> b,
		const float* x_init,
		const float* A_data,
		const int* A_rowptr,
		const int* A_colptr,
		float* r
	) {
		//The block that this thread is for
		const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (row_idx >= b.Size()) return;

		//Compute SparseMV
		const auto spmv = BinBlockCSR<BlockDim>::SparseMV(A_data, A_colptr, A_rowptr, x_init, row_idx);

		//Store to global result
		r[row_idx] = b[row_idx] - spmv;
	}

	template<int BlockDim>
    __global__ void initializeResidualKernel(
        const DeviceArrayView<float> b,
        cudaTextureObject_t x_init,
        const float* A_data,
		const int* A_rowptr,
		const int* A_colptr,
        float* r
    ) {
        //The block that this thread is for
        const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (row_idx >= b.Size()) return;

        //Compute SparseMV
        const auto spmv = BinBlockCSR<BlockDim>::SparseMV(A_data, A_colptr, A_rowptr, x_init, row_idx);

        //Store to global result
        r[row_idx] = b[row_idx] - spmv;
    }

} // namespace device
} // namespace surfelwarp


template<int BlockDim>
void surfelwarp::ApplySpMVBinBlockCSR<BlockDim>::ApplySpMV(DeviceArrayView<float> x, DeviceArraySlice<float> spmv_x, cudaStream_t stream) {
	//Sanity check
	SURFELWARP_CHECK_EQ(x.Size(), matrix_size_);
	SURFELWARP_CHECK_EQ(spmv_x.Size(), matrix_size_);

	//Perform spmv
	dim3 spmv_blk(128);
	dim3 spmv_grid(divUp(x.Size(), spmv_blk.x));
	device::performSparseMVKernel<BlockDim><<<spmv_grid, spmv_blk, 0, stream>>>(
		A_data_, A_rowptr_, A_colptr_, 
		x.RawPtr(), 
		spmv_x
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	//cudaSafeCall(cudaStreamSynchronize(stream));
	//cudaSafeCall(cudaGetLastError());
#endif
}


template<int BlockDim>
void surfelwarp::ApplySpMVBinBlockCSR<BlockDim>::ApplySpMVTextured(
	cudaTextureObject_t x,
	DeviceArraySlice<float> spmv_x,
	cudaStream_t stream
) {
	//simple sanity check
	SURFELWARP_CHECK_EQ(spmv_x.Size(), matrix_size_);

	dim3 spmv_blk(128);
	dim3 spmv_grid(divUp(spmv_x.Size(), spmv_blk.x));
	device::performSparseMVKernel<BlockDim><<<spmv_grid, spmv_blk, 0, stream>>>(
		A_data_, A_rowptr_, A_colptr_, 
		x, 
		spmv_x
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


template<int BlockDim>
void surfelwarp::ApplySpMVBinBlockCSR<BlockDim>::InitResidual(
	DeviceArrayView<float> x_init, 
	DeviceArrayView<float> b, 
	DeviceArraySlice<float> residual, 
	cudaStream_t stream
) {
	//Sanity check
	SURFELWARP_CHECK_EQ(x_init.Size(), matrix_size_);
	SURFELWARP_CHECK_EQ(b.Size(), matrix_size_);
	SURFELWARP_CHECK_EQ(residual.Size(), matrix_size_);

	dim3 spmv_blk(128);
    dim3 spmv_grid(divUp(b.Size(), spmv_blk.x));
	device::initializeResidualKernel<BlockDim><<<spmv_grid, spmv_blk, 0, stream>>>(
		b, 
		x_init,
		A_data_, A_rowptr_, A_colptr_, 
		residual
	);


	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

template<int BlockDim>
void surfelwarp::ApplySpMVBinBlockCSR<BlockDim>::InitResidualTextured(
	cudaTextureObject_t x_init,
	DeviceArrayView<float> b, 
	DeviceArraySlice<float> residual,
	cudaStream_t stream
) {
	//Sanity check
	SURFELWARP_CHECK_EQ(b.Size(), matrix_size_);
	SURFELWARP_CHECK_EQ(residual.Size(), matrix_size_);

	dim3 spmv_blk(128);
    dim3 spmv_grid(divUp(b.Size(), spmv_blk.x));
	device::initializeResidualKernel<BlockDim><<<spmv_grid, spmv_blk, 0, stream>>>(
		b, 
		x_init,
		A_data_, A_rowptr_, A_colptr_, 
		residual
	);


	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}