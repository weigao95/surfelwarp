#pragma once

#include "common/common_texture_utils.h"
#include "common/safe_call_utils.hpp"
#include "pcg_solver/BlockPCG.h"
#include "pcg_solver/BinBlockCSR.h"
#include "pcg_solver/solver_configs.h"
#include "math/DenseGaussian.h"

#include <device_launch_parameters.h>
#include <iostream>

namespace surfelwarp { namespace device {

	/**
	 * \brief lhs <- diag_blks rhs. Parallel over blocks
	 */
	template<int BlockDim>
	__global__ void blockDenseMVKernel(
		const float* diag_blks,
		const float* rhs,
		const unsigned num_blks,
		float* lhs
	) {
		//The block for this element
		const auto blk_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(blk_idx >= num_blks) return;
		
		//Load the rhs element
		float r_blk[BlockDim];
#pragma unroll
		for (auto j = 0; j < BlockDim; j++) {
			r_blk[j] = rhs[BlockDim * blk_idx + j];
		}

		//Compute the matrix vector product
		float l_blk[BlockDim];
		for (auto j = 0; j < BlockDim; j++)
		{
			//The current row index
			const auto row_idx = blk_idx * BlockDim + j;

			//Compute the matrix-vector product
			float l_row = 0.0f;
			for (auto i = 0; i < BlockDim; i++) {
				const auto mat_value = diag_blks[BlockDim * row_idx + i];
				l_row += mat_value * r_blk[i];
			}

			//Store the value locally
			l_blk[j] = l_row;
		}

		//Store the elements
#pragma unroll
		for (auto j = 0; j < BlockDim; j++) {
			lhs[BlockDim * blk_idx + j] = l_blk[j];
		}
	}

	/**
	 * \brief r <- b; x <- 0; d <- inv_diag r. This method parallelize over blocks
	 * \tparam BlockDim 
	 */
	template<int BlockDim>
	__global__ void blockPCGZeroInitKernel(
		const DeviceArrayView<float> b,
		const float* inv_diag_blks,
		float* r,
		float* d,
		float* x
	) {
		//Obtain the index for blocks
		const auto blk_idx = threadIdx.x + blockDim.x * blockIdx.x;
		const auto num_blks = b.Size() / BlockDim;
		if (blk_idx >= num_blks) return;

		//Load the b element
		float b_blk[BlockDim];
#pragma unroll
		for(auto j = 0; j < BlockDim; j++) {
			b_blk[j] = b[BlockDim * blk_idx + j];
		}

		//For each row in this block
		float d_local[BlockDim];
		for(auto j = 0; j < BlockDim; j++)
		{
			//The current row index
			const auto row_idx = blk_idx * BlockDim + j;

			//Compute the matrix-vector product
			float d_row = 0.0f;
			for(auto i = 0; i < BlockDim; i++) {
				const auto mat_value = inv_diag_blks[BlockDim * row_idx + i];
				d_row += mat_value * b_blk[i];
			}

			//Store the value
			d_local[j] = d_row;
		}

		//Store the element
		for(auto j = 0; j < BlockDim; j++)
		{
			const auto row_idx = blk_idx * BlockDim + j;
			r[row_idx] = b_blk[j];
			d[row_idx] = d_local[j];
			x[row_idx] = 0.0f;
		}
	}
		

    /**
     * \brief The second kernel in Weber et al 2013
     *        alpha <- (*delta_new) / (*dot_dq);
     *        x <- x + alpha d; t <- r - alpha q
     *        s <- inv_diag t; delta_old = delta_new
     */
	template<int BlockDim>
	__global__ void blockPCGSecondUpdateKernel(
		const DeviceArrayView<float> d,
		const float* q,
		const float* r,
		const float* inv_diag_blks,
		const float* delta_new,
		const float* dot_dq,
		float* x,
		float* s,
		float* t,
		float* delta_old
	) {
		//Obtain the index
		const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		const auto blk_idx = row_idx / BlockDim;
		const float alpha = (*delta_new) / (*dot_dq);
		if(row_idx >= d.Size()) return;

		//Load the r block, perform r <- r - alpha q
		float t_blk[BlockDim];
		for(auto i = 0; i < BlockDim; i++){
			t_blk[i] = r[blk_idx * BlockDim + i] - alpha * q[blk_idx * BlockDim + i];
		}

		//Perform s <- inv_diag * r
		float s_row = 0.0f;
		for(auto j = 0; j < BlockDim; j++){
			const auto mat_value = inv_diag_blks[BlockDim * row_idx + j];
			s_row += mat_value * t_blk[j];
		}

		// x <- x + alpha d; store the value
		x[row_idx] += alpha * d[row_idx];
		t[row_idx] = t_blk[(row_idx % BlockDim)];
		s[row_idx] = s_row;

		//delta_old_ = delta_new_
		if (row_idx == 0) {
			*delta_old = (*delta_new);
		}
	}


	/**
     * \brief beta <- delta_new / delta_old; d <- s + beta d
     */
    template<int BlockDim>
	__global__ void blockPCGThirdUpdateKernel(
		const DeviceArrayView<float> r,
		const float* delta_old,
		const float* delta_new,
		float* d
	) {
		//Check the size
		const auto row_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(row_idx >= r.Size()) return;

		//Compute the delta
		const float beta = (*delta_new) / (*delta_old);
		d[row_idx] = r[row_idx] + beta * d[row_idx];
	}

} /* End of namespace device */ }; /* End of namespace surfelwarp */


template<int BlockDim>
surfelwarp::BlockPCG<BlockDim>::BlockPCG(size_t max_matrix_size, cudaStream_t stream) {
    //Allocate the buffer
    m_max_matrix_size = 0;
    allocateBuffer(max_matrix_size);
	m_spmv_handler = nullptr;

    //Initialize the cuda and cublas resource
	m_stream = stream;
    cublasSafeCall(cublasCreate(&m_cublas_handle));
	cublasSafeCall(cublasSetPointerMode(m_cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
    cublasSafeCall(cublasSetStream(m_cublas_handle, m_stream));
}

template<int BlockDim>
void surfelwarp::BlockPCG<BlockDim>::UpdateCudaStream(cudaStream_t stream) {
    m_stream = stream;
    cublasSafeCall(cublasSetStream(m_cublas_handle, m_stream));
}

template<int BlockDim>
surfelwarp::BlockPCG<BlockDim>::~BlockPCG() {
    releaseBuffer();
	cublasDestroy(m_cublas_handle);
}

template<int BlockDim>
void surfelwarp::BlockPCG<BlockDim>::allocateBuffer(size_t max_matrix_size) {
    //Rectify the matrix_size to the multiple of BlockSize
    const auto max_blk_num = divUp(max_matrix_size, BlockDim);
    const auto rectify_matrix_size = max_blk_num * BlockDim;

    //Do not need allocate again
    if(m_max_matrix_size >= rectify_matrix_size) return;

    //Release the buffer first if already allocated
    if(m_max_matrix_size > 0) releaseBuffer();

    //Initialize the size
    m_max_matrix_size = rectify_matrix_size;

    //Allocate the buffer
    r_.AllocateBuffer(rectify_matrix_size);
	t_.AllocateBuffer(rectify_matrix_size);
    d_.AllocateBuffer(rectify_matrix_size);
	q_.AllocateBuffer(rectify_matrix_size);
	s_.AllocateBuffer(rectify_matrix_size);
    
    //Explicit malloc on device: need to release them
    cudaSafeCall(cudaMalloc((void**)(&delta_old_), sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)(&delta_new_), sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)(&dot_dq_), sizeof(float)));

	//Allocate the page-locked memory for converge checking
	cudaSafeCall(cudaMallocHost((void**)(&delta_0_pagelock_), sizeof(float)));
	cudaSafeCall(cudaMallocHost((void**)(&delta_pagelock_), sizeof(float)));

	//Create the texture for sparse mv
    d_texture_ = create1DLinearTexture(d_);
}

template<int BlockDim>
void surfelwarp::BlockPCG<BlockDim>::releaseBuffer() {
    //Zero the size of this solver
    m_max_matrix_size = 0;

    //Release the buffer maintained by device array
    r_.ReleaseBuffer();
	t_.ReleaseBuffer();
	d_.ReleaseBuffer();
	q_.ReleaseBuffer();
	s_.ReleaseBuffer();

    //Release the explicit malloced buffer
    cudaSafeCall(cudaFree(delta_old_));
    cudaSafeCall(cudaFree(delta_new_));
    cudaSafeCall(cudaFree(dot_dq_));
    cudaSafeCall(cudaFreeHost(delta_pagelock_));
    cudaSafeCall(cudaFreeHost(delta_0_pagelock_));

	//Destroy the texture
	cudaDestroyTextureObject(d_texture_);
}


template<int BlockDim>
bool surfelwarp::BlockPCG<BlockDim>::SetSolverInput(
	DeviceArrayView<float> inv_diag_blks,
	typename ApplySpMVBase<BlockDim>::Ptr spmv_handler,
	DeviceArrayView<float> b,
	DeviceArraySlice<float> x_init,
	size_t actual_size
) {
	//Determine the size of the matrix
	if(actual_size == 0) actual_size = b.Size();
	
	//Check the size of allcoated buffer
	if(m_max_matrix_size < actual_size) return false;
	
	//This size can be solved given current buffer
	m_actual_matrix_size = actual_size;
	
	//Simple sanity check
	const auto num_blks = divUp(m_actual_matrix_size, BlockDim);
	SURFELWARP_CHECK(actual_size % BlockDim == 0);
	SURFELWARP_CHECK_EQ(inv_diag_blks.Size(), BlockDim * BlockDim * num_blks);
	SURFELWARP_CHECK_EQ(b.Size(), BlockDim * num_blks);
	SURFELWARP_CHECK_EQ(x_init.Size(), BlockDim * num_blks);
	
	//Seems correct
	m_inv_diagonal_blks = inv_diag_blks;
	r_.ResizeArrayOrException(actual_size);
	t_.ResizeArrayOrException(actual_size);
	d_.ResizeArrayOrException(actual_size);
	q_.ResizeArrayOrException(actual_size);
	s_.ResizeArrayOrException(actual_size);
	
	m_spmv_handler = spmv_handler;
	b_ = b;
	x_ = x_init;
	
	return true;
}

template<int BlockDim>
surfelwarp::DeviceArrayView<float> surfelwarp::BlockPCG<BlockDim>::Solve(size_t max_iterations, bool zero_init) {
	return SolveNoConvergeCheck(max_iterations, zero_init);
}

template<int BlockDim>
surfelwarp::DeviceArrayView<float> surfelwarp::BlockPCG<BlockDim>::SolveNoConvergeCheck(size_t max_iteration, bool zero_init) {
	//Do initialize
    if(zero_init) {
        ZeroIntialize();
    } else {
        Initialize(x_.ArrayView());
    }

    //The main loop
    for(auto i = 0; i < max_iteration; i++) {
        PerformSparseMV();
		//TexturedSparseMV();
        PerformSecondUpdate();
        PerformThirdUpdate();
    }

    return x_.ArrayView();
}


template<int BlockDim>
surfelwarp::DeviceArrayView<float> surfelwarp::BlockPCG<BlockDim>::SolveConvergeChecked(
        size_t max_iteration,
        bool zero_init,
        float epsilon
) {
    //Same buffer, for different purpose
    DeviceArray<float> x_init = DeviceArray<float>(x_.RawPtr(), m_actual_matrix_size);

    //Do initialize
    if(zero_init) {
        ZeroIntialize();
    } else {
        Initialize(x_.ArrayView());
    }

    //Download delta_0 for convergence check
    //float delta_0 = 0.0f, delta = 0.0f;
    const float eps_square = epsilon * epsilon;
    cudaSafeCall(cudaMemcpyAsync(delta_0_pagelock_, delta_new_, sizeof(float), cudaMemcpyDeviceToHost, m_stream));


    //The main loop
    for(auto i = 0; i < max_iteration; i++) {
        PerformSparseMV();
		//TexturedSparseMV();
        PerformSecondUpdate();
        PerformThirdUpdate();

        //Check converge
        cudaSafeCall(cudaMemcpyAsync(delta_pagelock_, delta_new_, sizeof(float), cudaMemcpyDeviceToHost, m_stream));
        cudaSafeCall(cudaStreamSynchronize(m_stream));
        if(std::abs(*delta_pagelock_) < eps_square * std::abs(*delta_0_pagelock_)) {
            break;
        }
    }

    return x_.ArrayView();
}



template<int BlockDim>
void surfelwarp::BlockPCG<BlockDim>::Initialize(const DeviceArrayView<float>& x_init) {
	//Use handler for residual initialization
	m_spmv_handler->InitResidual(x_init, b_, r_.ArraySlice(), m_stream);

    //Use the pre-conditioner to intialize d
    const auto num_blocks = m_actual_matrix_size / BlockDim;
    dim3 init_blk(64);
    dim3 init_grid(divUp(num_blocks, init_blk.x));
    device::blockDenseMVKernel<BlockDim><<<init_grid, init_blk, 0, m_stream>>>(m_inv_diagonal_blks, r_, num_blocks, d_);

    //Perform dot-product using cublas
    cublasSdot(m_cublas_handle, r_.ArraySize(), r_.Ptr(), 1, d_.Ptr(), 1, delta_new_);
}

template<int BlockDim>
void surfelwarp::BlockPCG<BlockDim>::ZeroIntialize() {
    const auto num_blocks = m_actual_matrix_size / BlockDim;
    dim3 init_blk(64);
    dim3 init_grid(divUp(num_blocks, init_blk.x));
    device::blockPCGZeroInitKernel<BlockDim><<<init_grid, init_blk, 0, m_stream>>>(b_, m_inv_diagonal_blks, r_, d_, x_);

    //Perform dot-product using cublas
    cublasSdot(m_cublas_handle, r_.ArraySize(), r_.Ptr(), 1, d_.Ptr(), 1, delta_new_);
}

template<int BlockDim>
void surfelwarp::BlockPCG<BlockDim>::PerformSparseMV() {
	//Using the handler
	m_spmv_handler->ApplySpMV(d_.ArrayView(), q_.ArraySlice(), m_stream);

    //dot_dq <- dot(d, q)
    cublasSdot(m_cublas_handle, d_.ArraySize(), q_.Ptr(), 1, d_.Ptr(), 1, dot_dq_);
}

template<int BlockDim>
void surfelwarp::BlockPCG<BlockDim>::TexturedSparseMV() {
	//Use the handler
	m_spmv_handler->ApplySpMVTextured(d_texture_, q_.ArraySlice(), m_stream);

	//dot_dq <- dot(d, q)
	cublasSdot(m_cublas_handle, d_.ArraySize(), q_.Ptr(), 1, d_.Ptr(), 1, dot_dq_);
}



template<int BlockDim>
void surfelwarp::BlockPCG<BlockDim>::PerformSecondUpdate() {
    dim3 update_blk(128);
    dim3 update_grid(divUp(m_actual_matrix_size, update_blk.x));
    device::blockPCGSecondUpdateKernel<BlockDim><<<update_grid, update_blk, 0, m_stream>>>(
		d_.ArrayView(),
        q_,
		r_,
        m_inv_diagonal_blks,
        delta_new_,
        dot_dq_,
        x_,
        s_,
		t_,
		delta_old_
    );

    //delta_new <- dot(r, s)
    cublasSdot(m_cublas_handle, t_.ArraySize(), t_.Ptr(), 1, s_.Ptr(), 1, delta_new_);
}

template<int BlockDim>
void surfelwarp::BlockPCG<BlockDim>::PerformThirdUpdate() {
    dim3 update_blk(128);
    dim3 update_grid(divUp(m_actual_matrix_size, update_blk.x));
    device::blockPCGThirdUpdateKernel<BlockDim><<<update_grid, update_blk, 0, m_stream>>>(
		s_.ArrayView(),
        delta_old_, delta_new_,
        d_
    );
	r_.swap(t_);
}



