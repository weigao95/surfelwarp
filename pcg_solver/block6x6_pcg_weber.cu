#include "common/common_types.h"
#include "common/common_utils.h"
#include "common/sanity_check.h"
#include "common/device_intrinsics.h"
#include "math/DenseGaussian.h"
#include "math/DenseLDLT.h"
#include "pcg_solver/solver_configs.h"
#include "pcg_solver/block6x6_pcg_weber.h"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include <cub/cub.cuh>

#include <iostream>

namespace surfelwarp { namespace device {

	/**
	 * \brief Perform parallel matrix inverse on 6x6 psd matrix array
	 * \tparam num_threads Each thread process a matrix
	 * \param A Input matrix array, will not be touched
	 * \param A_inversed The output matrix array
	 * \param num_matrix 
	 */
	template <int num_threads = 64>
	__global__ void matrix6x6InverseKernel(
		const float* A,
		float* A_inversed,
		int num_matrix
	) {
		//Load the matrix into the shared memory
		__shared__ float factored_matrix[36 * num_threads];
		__shared__ float inversed_matrix[36 * num_threads];

		//The input matrix pointer for this block
		const int blk_matrix_offset = 36 * blockDim.x * blockIdx.x;
		const float* A_this_blk = A + blk_matrix_offset;

		//Cooperative loading
		for (auto k = 0; k < 36; k++) { // There are 36 x num_threads float need to be loaded
			if(blk_matrix_offset + k * num_threads + threadIdx.x < num_matrix * 36)
				factored_matrix[k * num_threads + threadIdx.x] = A_this_blk[k * num_threads + threadIdx.x]; //Each thread loads one element
		}

		//Sync here
		__syncthreads();

		//Call the Gaussian inversion
		float* A_this_thread = &(factored_matrix[36 * threadIdx.x]);
		float* A_inv_this_thread = &(inversed_matrix[36 * threadIdx.x]);
		DenseGaussian<6>::Inverse(A_this_thread, A_inv_this_thread);

		//Sync again
		__syncthreads();

		//Cooperative storing
		float* A_inv_this_blk = A_inversed + blk_matrix_offset;
		for (auto k = 0; k < 36; k++) { // There are 36 x num_threads float need to be loaded
			if (blk_matrix_offset + k * num_threads + threadIdx.x < num_matrix * 36)
				A_inv_this_blk[k * num_threads + threadIdx.x] = inversed_matrix[k * num_threads + threadIdx.x]; //Each thread stores one element
		}
	}


	__device__ float nu_old_blk6x6;
	__device__ float nu_new_blk6x6;
	__device__ float reduce_partials_blk6x6[max_reduce_blocks]; //The maximum number of blocks to perform reduce for dot(a, b)
	

	/**
	 * \brief r <- b; s <- inv_diag_blks * b; mu_new <- dot(r, s)
	 * \tparam num_warps The FIXED number of warps in this kernel, for reduction
	 * \param b 
	 * \param inv_diag_blks 
	 * \param r 
	 * \param s 
	 */
	template <int num_warps = reduce_block_warps>
	__global__ void block6x6InitKernel(
		const PtrSz<const float> b,
		const PtrSz<const float> inv_diag_blks,
		PtrSz<float> r, 
		PtrSz<float> s,
		PtrSz<float> x
	) {
		//r <- b; s <- inv_diag_blks * b;
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;

		//The dot product from this row for mu_new <- dot(r, s)
		float dot_this_row = 0.0f;
		if (idx < b.size) {
			const int blk_idx = idx / 6;

			//Perform the block matrix vector product
			float s_row = 0.0f;
			for (auto j = 0; j < 6; j++) {
				const float mat_value = inv_diag_blks[6 * idx + j];
				const float b_value = b[6 * blk_idx + j];
				s_row += mat_value * b_value;
			}
			const float r_row = b[idx];
			dot_this_row = s_row * r_row;

			//Store the value to s and r
			s[idx] = s_row;
			r[idx] = r_row;
			x[idx] = 0.0f;
		}

		//Warp reduction on dot_this_row
		const int warp_id = threadIdx.x >> 5;
		const int lane_id = threadIdx.x & 31;
		float scanned_dot = dot_this_row;
		scanned_dot = warp_scan(scanned_dot);

		//Store the reduced warp_dot to shared memory for block scan
		__shared__ float warp_dot[num_warps];
		if (lane_id == 31) warp_dot[warp_id] = scanned_dot;

		//Perform reduct on the warp_dot
		__syncthreads();
		if (warp_id == 0) {
			float warp_dot_reduce = 0.0f;
			if (lane_id < num_warps)
				warp_dot_reduce = warp_dot[lane_id];
			//Do warp scan again
            warp_dot_reduce = warp_scan(warp_dot_reduce);
			//Store to global memory
			if (lane_id == 31) reduce_partials_blk6x6[blockIdx.x] = warp_dot_reduce;
		}
	}


	__global__ void block6x6ReducePartialKernel() {
		float sum = 0.0f;
		if (threadIdx.x < num_reduce_blocks_6x6) {
			sum = reduce_partials_blk6x6[threadIdx.x];
		}

		sum = warp_scan(sum);
		if (threadIdx.x == 31) {
			nu_new_blk6x6 = sum; // nu_new <- dot(r, s)
		}
	}


	/* nu_old <- nu_new; q <- A s; alpha <- nu_new / dot(q, s); */
    template<int num_warps = reduce_block_warps>
	__global__ void block6x6PCGKernel_0(
		const PtrSz<const float> A_data,
		const PtrSz<const int> A_colptr,
		const PtrSz<const int> A_rowptr,
		const PtrSz<const float> s,
		PtrSz<float> q
	) {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if(idx == 0){
            nu_old_blk6x6 = nu_new_blk6x6;
        }

		const int warp_id = threadIdx.x >> 5;
		const int lane_id	= threadIdx.x & 31;
		float dot_this_row = 0;

		//Perform a sparse matrix-vector product
		if(idx < s.size) {
			int begin = A_rowptr[idx];
			const int end = A_rowptr[idx + bin_size];
			int column_offset = (begin - lane_id) / 6 + lane_id;
			float sp_mv = 0.0f;
			while (begin < end) {
				const int colume = A_colptr[column_offset];
				for(auto j = 0; j < 6; j++){
					float mat_data = A_data[begin];
					float s_data = colume >= 0 ? s[colume + j] : 0;
					sp_mv += mat_data * s_data;
					begin += bin_size;
				}

				//Increase the column index
				column_offset += bin_size;
			}

            //The value of this row
            q[idx] = sp_mv;
            dot_this_row = sp_mv * s[idx];
		}

        //Perform warp scan
        float scanned_dot = dot_this_row;
        scanned_dot = warp_scan(scanned_dot);

        //Store the reduced warp_dot to shared memory for block scan
        __shared__ float warp_dot[num_warps];
        if (lane_id == 31) warp_dot[warp_id] = scanned_dot;

        //Perform reduct on the warp_dot
        __syncthreads();
        if (warp_id == 0) {
            float warp_dot_reduce = 0.0f;
            if (lane_id < num_warps)
                warp_dot_reduce = warp_dot[lane_id];
            //Do warp scan again
            warp_dot_reduce = warp_scan(warp_dot_reduce);
            //Store to global memory
            if (lane_id == 31) reduce_partials_blk6x6[blockIdx.x] = warp_dot_reduce;
        }
	}


	template<int num_warps = reduce_block_warps>
	__global__ void block6x6PCGKernel_0(
		const PtrSz<const float> A_data,
		const PtrSz<const int> A_colptr,
		const PtrSz<const int> A_rowptr,
		cudaTextureObject_t s,
		PtrSz<float> q
	) {
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx == 0) {
			nu_old_blk6x6 = nu_new_blk6x6;
		}

		const int warp_id = threadIdx.x >> 5;
		const int lane_id = threadIdx.x & 31;
		float dot_this_row = 0;

		//Perform a sparse matrix-vector product
		if (idx < q.size) {
			int begin = A_rowptr[idx];
			const int end = A_rowptr[idx + bin_size];
			int column_offset = (begin - lane_id) / 6 + lane_id;
			float sp_mv = 0.0f;
			while (begin < end) {
				const int colume = A_colptr[column_offset];
				for (auto j = 0; j < 6; j++) {
					const float mat_data = A_data[begin];
					const float s_data = (colume >= 0) ? fetch1DLinear<float>(s, colume + j) : 0.0f;
					sp_mv += mat_data * s_data;
					begin += bin_size;
				}

				//Increase the column index
				column_offset += bin_size;
			}

			//The value of this row
			q[idx] = sp_mv;
			dot_this_row = sp_mv * fetch1DLinear<float>(s, idx);
		}

		//Perform warp scan
		float scanned_dot = dot_this_row;
		scanned_dot = warp_scan(scanned_dot);

		//Store the reduced warp_dot to shared memory for block scan
		__shared__ float warp_dot[num_warps];
		if (lane_id == 31) warp_dot[warp_id] = scanned_dot;

		//Perform reduct on the warp_dot
		__syncthreads();
		if (warp_id == 0) {
			float warp_dot_reduce = 0.0f;
			if (lane_id < num_warps)
				warp_dot_reduce = warp_dot[lane_id];
			//Do warp scan again
			warp_dot_reduce = warp_scan(warp_dot_reduce);
			//Store to global memory
			if (lane_id == 31) reduce_partials_blk6x6[blockIdx.x] = warp_dot_reduce;
		}
	}

	/**
	 * \brief alpha <- nu_new / dot(q, s); x <- x + alpha * s; 
	 *        t <- r - alpha * q; p <- M_inv*t; nu_new <- dot(t, p)
	 * \tparam num_warps The FIXED number of warps in this kernel
	 */
	template<int num_warps = reduce_block_warps>
	__global__ void block6x6PCGKernel_1(
		const PtrSz<const float> s,
		const PtrSz<const float> r,
        const PtrSz<const float> q,
		const PtrSz<const float> inv_diag_blks,
		PtrSz<float> x, 
		PtrSz<float> t,
		PtrSz<float> p
	) {
		//Each block performs a reduction for alpha = dot(q, s)
		__shared__ float alpha;
		const int warp_id = threadIdx.x >> 5;
		const int lane_id = threadIdx.x & 31;
        float scanned_dot;

		//Perform reduction on warp_0
		if (warp_id == 0) {
			scanned_dot = 0.0f;
			if (lane_id < num_reduce_blocks_6x6) {
				scanned_dot = reduce_partials_blk6x6[lane_id];
			}
            scanned_dot = warp_scan(scanned_dot);
			if (lane_id == 31) {
                alpha = nu_new_blk6x6 / scanned_dot;
            }
		}

		//Do sync to broadcast alpha
		__syncthreads();
		const float alpha_thread = alpha;

        //float alpha_thread = alpha;
        const int idx = threadIdx.x + blockDim.x * blockIdx.x;
        float dot_this_row = 0.0f;
        if (idx < x.size) {
            const int blk_idx = idx / 6;

            //Block matrix vector product
            float p_row = 0.0;
            float mat_value, r_value;
            for(auto j = 0; j < 6; j++) {
                mat_value = inv_diag_blks[6 * idx + j];
                r_value = r[6 * blk_idx + j] - alpha_thread * q[6 * blk_idx + j];
                p_row += mat_value * r_value;
            }
            p[idx] = p_row; //p <- M_inv * r

            //float r_row = r[idx];
            //float q_row = q[idx];
            const float r_row_new = r[idx] - alpha_thread * q[idx];
            t[idx] = r_row_new; // t <- r - alpha * q
            x[idx] += alpha_thread * s[idx]; // x <- x + alpha s
            dot_this_row = p_row * r_row_new;
        }

        //Perform in block reduction on dot(q, s)
        scanned_dot = dot_this_row;
        scanned_dot = warp_scan(scanned_dot);

        //Store the reduced warp_dot to shared memory for block scan
        __shared__ float warp_dot[num_warps];
        if (lane_id == 31) warp_dot[warp_id] = scanned_dot;

        __syncthreads();
        if (warp_id == 0) {
            float warp_dot_reduce = 0.0f;
            if (lane_id < num_warps) {
                warp_dot_reduce = warp_dot[lane_id];
            }
            //Do warp scan again
            warp_dot_reduce = warp_scan(warp_dot_reduce);

            //Store to global memory
            if (lane_id == 31) reduce_partials_blk6x6[blockIdx.x] = warp_dot_reduce;
        }
	}


	


	/**
	 * \brief nu_new <- dot(t, p); beta <- nu_new/nu_old; s <- p + beta s
	 */
	__global__ void block6x6PCGKernel_2(
		const PtrSz<const float> p,
		PtrSz<float> s
	) {
		//Each block perform a reduce to compute beta
		__shared__ float beta;
		const int warp_id = threadIdx.x >> 5;
		const int lane_id = threadIdx.x & 31;

		if(warp_id == 0)
		{
			float dot_reduce = 0.0f;
			if(lane_id < num_reduce_blocks_6x6) {
				dot_reduce = reduce_partials_blk6x6[lane_id];
			}
			dot_reduce = warp_scan(dot_reduce);
			if(lane_id == 31) {
                if(blockIdx.x == 0) nu_new_blk6x6 = dot_reduce;
				beta = dot_reduce / nu_old_blk6x6;

                //Debug code: seems correct
                //printf("Beta from device %f \n", beta);
			}
		}

		//Do sync to broadcast the value of beta
		__syncthreads();
		const float beta_thread = beta;
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < p.size) {
			s[idx] = p[idx] + beta_thread * s[idx];
		}
	}




}; /* End of namespace device */ }; /* End of namespace surfelwarp */



void surfelwarp::block6x6_pcg_weber(
	const DeviceArray<float>& diag_blks, 
	const DeviceArray<float>& A_data,
	const DeviceArray<int>& A_colptr,
	const DeviceArray<int>& A_rowptr, 
	const DeviceArray<float>& b, 
	DeviceArray<float>& x_buffer, 
	DeviceArray<float>& inv_diag_blk_buffer,
	DeviceArray<float>& p_buffer, 
	DeviceArray<float>& q_buffer, 
	DeviceArray<float>& r_buffer, 
	DeviceArray<float>& s_buffer, 
	DeviceArray<float>& t_buffer, 
	DeviceArray<float>& valid_x, 
	int max_iters, 
	cudaStream_t stream
) {
	
	//Correct the size of array
	size_t N = b.size();
	DeviceArray<float> inv_diag_blks = DeviceArray<float>(inv_diag_blk_buffer.ptr(), diag_blks.size());
	valid_x = DeviceArray<float>(x_buffer.ptr(), N);
	DeviceArray<float> p = DeviceArray<float>(p_buffer.ptr(), N);
	DeviceArray<float> q = DeviceArray<float>(q_buffer.ptr(), N);
	DeviceArray<float> r = DeviceArray<float>(r_buffer.ptr(), N);
	DeviceArray<float> s = DeviceArray<float>(s_buffer.ptr(), N);
	DeviceArray<float> t = DeviceArray<float>(t_buffer.ptr(), N);

	//Compute the inverse of diag blocks for pre-conditioning
	cudaSafeCall(cudaMemsetAsync(valid_x.ptr(), 0, sizeof(float) * valid_x.size(), stream));
	block6x6_diag_inverse(diag_blks, inv_diag_blks, N / 6, stream);

    //The init kernel
    block6x6_init_kernel(b, inv_diag_blks, r, s, valid_x, stream);

    //The main loop
    for(auto i = 0; i < max_iters; i++) {
        block6x6_pcg_kernel_0(A_data, A_colptr, A_rowptr, s, q, stream);
        block6x6_pcg_kernel_1(s, r, q, inv_diag_blks, valid_x, t, p, stream);
        block6x6_pcg_kernel_2(p, s, stream);
        r.swap(t);
    }

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}



void surfelwarp::block6x6_pcg_weber(
	const DeviceArray<float>& diag_blks, 
	const DeviceArray<float>& A_data, 
	const DeviceArray<int>& A_colptr, 
	const DeviceArray<int>& A_rowptr,
	const DeviceArray<float>& b,
	DeviceArray<float>& x_buffer, 
	DeviceArray<float>& inv_diag_blk_buffer, 
	DeviceArray<float>& p_buffer,
	DeviceArray<float>& q_buffer, 
	DeviceArray<float>& r_buffer, 
	DeviceArray<float>& s_buffer, 
	cudaTextureObject_t s_texture,
	DeviceArray<float>& t_buffer, 
	DeviceArray<float>& valid_x, 
	int max_iters, 
	cudaStream_t stream
) {
	//Correct the size of array
	size_t N = b.size();
	DeviceArray<float> inv_diag_blks = DeviceArray<float>(inv_diag_blk_buffer.ptr(), diag_blks.size());
	valid_x = DeviceArray<float>(x_buffer.ptr(), N);
	DeviceArray<float> p = DeviceArray<float>(p_buffer.ptr(), N);
	DeviceArray<float> q = DeviceArray<float>(q_buffer.ptr(), N);
	DeviceArray<float> r = DeviceArray<float>(r_buffer.ptr(), N);
	DeviceArray<float> s = DeviceArray<float>(s_buffer.ptr(), N);
	DeviceArray<float> t = DeviceArray<float>(t_buffer.ptr(), N);

	//Compute the inverse of diag blocks for pre-conditioning
	block6x6_diag_inverse(diag_blks, inv_diag_blks, N / 6, stream);

	//The init kernel
	block6x6_init_kernel(b, inv_diag_blks, r, s, valid_x, stream);

	//The main loop
	for (auto i = 0; i < max_iters; i++) {
		block6x6_pcg_kernel_0(A_data, A_colptr, A_rowptr, s_texture, q, stream);
		block6x6_pcg_kernel_1(s, r, q, inv_diag_blks, valid_x, t, p, stream);
		block6x6_pcg_kernel_2(p, s, stream);
		r.swap(t);
	}

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}




void surfelwarp::block6x6_diag_inverse(const float * A, float * A_inversed, int num_matrix, cudaStream_t stream)
{
	const int threads_per_blk = 64;
	dim3 blk(threads_per_blk);
	dim3 grid(divUp(num_matrix, blk.x));
	device::matrix6x6InverseKernel<threads_per_blk><<<grid, blk, 0, stream>>>(A, A_inversed, num_matrix);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

/* r <- b; s <- inv_diag_blks; mu_new <- dot(r, s)  */
void surfelwarp::block6x6_init_kernel(
	const DeviceArray<float>& b, 
	const DeviceArray<float>& inv_diag_blks, 
	DeviceArray<float>& r, 
	DeviceArray<float>& s,
	DeviceArray<float>& x,
	cudaStream_t stream
) {
    dim3 blk(reduce_block_threads);
    //dim3 grid(divUp(b.size(), blk.x));
    dim3 grid(num_reduce_blocks_6x6);
    device::block6x6InitKernel<<<grid, blk, 0, stream>>>(b, inv_diag_blks, r, s, x);

    //Perform a reduction on the global memory
    dim3 reduce_blk(32);
    dim3 reduce_grid(1);
    device::block6x6ReducePartialKernel<<<reduce_grid, reduce_blk, 0, stream>>>();

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


/* nu_old <- nu_new; q <- A s; alpha <- nu_old / dot(q, s); */
void surfelwarp::block6x6_pcg_kernel_0(
		const DeviceArray<float> &A_data,
		const DeviceArray<int> &A_colptr,
		const DeviceArray<int> &A_rowptr,
		const DeviceArray<float> &s,
		DeviceArray<float> &q, cudaStream_t stream
) {
	dim3 blk(reduce_block_threads);
	//dim3 grid(divUp(s.size(), blk.x));
    dim3 grid(num_reduce_blocks_6x6);
	device::block6x6PCGKernel_0<<<grid, blk, 0, stream>>>(A_data, A_colptr, A_rowptr, s, q);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::block6x6_pcg_kernel_0(
	const DeviceArray<float>& A_data, 
	const DeviceArray<int>& A_colptr,
	const DeviceArray<int>& A_rowptr, 
	cudaTextureObject_t s, 
	DeviceArray<float>& q, 
	cudaStream_t stream
) {
	dim3 blk(reduce_block_threads);
	//dim3 grid(divUp(s.size(), blk.x));
	dim3 grid(num_reduce_blocks_6x6);
	device::block6x6PCGKernel_0<<<grid, blk, 0, stream>>>(A_data, A_colptr, A_rowptr, s, q);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


/* alpha <- nu_new / dot(q, s); x <- x + alpha * s;
 * t <- r - alpha * q; p <- M_inv*t; nu_new <- dot(t, p) */
void surfelwarp::block6x6_pcg_kernel_1(
        const DeviceArray<float>& s,
        const DeviceArray<float>& r,
        const DeviceArray<float>& q,
        const DeviceArray<float>& inv_diag_blks,
        DeviceArray<float>& x,
        DeviceArray<float>& t,
        DeviceArray<float>& p,
        cudaStream_t stream
) {
    dim3 blk(reduce_block_threads);
    dim3 grid(num_reduce_blocks_6x6);
    device::block6x6PCGKernel_1<<<grid, blk, 0, stream>>>(s, r, q, inv_diag_blks, x, t, p);
}


void surfelwarp::block6x6_pcg_kernel_2(
	const DeviceArray<float>& p,
	DeviceArray<float>& s, 
	cudaStream_t stream
) {
    dim3 blk(256);
    dim3 grid(divUp(s.size(), blk.x));
    device::block6x6PCGKernel_2<<<grid, blk, 0, stream>>>(p, s);
}


/*
 * Below are the checking subroutines defined for 6x6 pcg solver
 */
void surfelwarp::checkBlock6x6Init(
        const std::vector<float> &b,
        const std::vector<float> &inv_diags,
        std::vector<float>& h_r,
        std::vector<float>& h_s
) {
    //Prepare the data
    DeviceArray<float> b_dev, d_inv_diags, r, s, x;
    b_dev.upload(b);
    d_inv_diags.upload(inv_diags);
    r.create(b_dev.size());
    s.create(b_dev.size());
	x.create(b_dev.size());

    //Call the function
    block6x6_init_kernel(b_dev, d_inv_diags, r, s, x);

    //Check the value of dot product
    cudaDeviceSynchronize();
    r.download(h_r); s.download(h_s);
    float dot_value = 0;
    for(auto i = 0;i < h_s.size();i++){
        dot_value += h_r[i] * h_s[i];
    }

    //Frist check r == b
    assert(h_r.size() == b.size());
    for(auto i = 0; i < b.size(); i++) {
        assert(std::abs(h_r[i] - b[i]) < 1e-4);
    }

    //Check s = inv_diag * b
    for(auto row = 0; row < b.size(); row++) {
        int blk_idx = row / 6;
        int inblk_offset = row % 6;
        int diag_offset = 36 * blk_idx;
        int diag_start_idx = diag_offset + 6 * inblk_offset;
        float s_row = 0.0f;
        for(auto j = 0; j < 6; j++) {
            s_row += inv_diags[diag_start_idx + j] * b[6 * blk_idx + j];
        }
        assert(std::abs(s_row - h_s[row]) < 1e-4);
    }

    //Compare it with device value
    float dot_device;
    cudaMemcpyFromSymbol(&dot_device, device::nu_new_blk6x6, sizeof(float), 0, cudaMemcpyDeviceToHost);
    if(std::abs((dot_device - dot_value) / dot_value) > 1e-6) {
        std::cout << "Relative err in init kernel dot product " << std::abs((dot_device - dot_value) / dot_value) << std::endl;
    }
}

void surfelwarp::checkBlock6x6Init(
        const std::vector<float> &b,
        const std::vector<float> &inv_diags
) {
    std::vector<float> r, s;
    checkBlock6x6Init(b, inv_diags, r, s);
}


void surfelwarp::checkBlock6x6Kernel_0(
        const std::vector<float> &A_data,
        const std::vector<int> &A_rowptr,
        const std::vector<int> &A_colptr,
        const std::vector<float> &s,
        //Output for later checking
        std::vector<float>& q_device
) {
    //Prepare the data
    DeviceArray<float> d_A_data, s_dev, q_dev;
    DeviceArray<int> d_A_rowptr, d_A_colptr;
    d_A_data.upload(A_data);
    s_dev.upload(s);
    q_dev.create(s.size());
    d_A_colptr.upload(A_colptr);
    d_A_rowptr.upload(A_rowptr);

    //Call device function
    block6x6_pcg_kernel_0(d_A_data, d_A_colptr, d_A_rowptr, s_dev, q_dev);

    //Perform matrix vector product on host
    const auto matrix_size = s.size();
    std::vector<float> q_host;
    hostEigenSpMV(A_data, A_rowptr, A_colptr, matrix_size, s, q_host);

    //Check q = A s
    q_device.clear();
    q_dev.download(q_device);
    float maximum_relative_err = 0.0f;
    assert(q_device.size() == q_host.size());
    for(auto i = 0; i < q_host.size(); i++) {
        float host_value = q_host[i];
        float device_value = q_device[i];
        if(std::abs(host_value - device_value) > 1e-4) {
            if(std::abs((host_value - device_value) / host_value) > maximum_relative_err) {
                maximum_relative_err = std::abs((host_value - device_value) / host_value);
            }
        }
    }
    std::cout << "The maximum relative error in SpMV " << maximum_relative_err << std::endl;

    //Next check the value of dot product
    float dev_dot_reduce[max_reduce_blocks];
    cudaMemcpyFromSymbol(dev_dot_reduce, device::reduce_partials_blk6x6, sizeof(float) * max_reduce_blocks, 0, cudaMemcpyDeviceToHost);
    float dev_dot = 0.0f;
    for(auto j = 0; j < num_reduce_blocks_6x6; j++) {
        dev_dot += dev_dot_reduce[j];
    }

    //Compute the dot prodcut at host
    float h_dot = 0.0f;
    for(auto j = 0; j < q_host.size(); j++) {
        h_dot += q_host[j] * s[j];
    }
    assert(std::abs((h_dot - dev_dot) / dev_dot) < 1e-4);
}



void surfelwarp::checkBlock6x6Kernel_1(
        const std::vector<float> &s,
        const std::vector<float> &r,
        const std::vector<float> &q,
        const std::vector<float> &inv_diag_blks,
        std::vector<float> &x,
        std::vector<float> &t,
        std::vector<float> &p
)
{
	//Prepare data for input
	DeviceArray<float> s_dev, r_dev, q_dev, inv_diag_blks_dev, x_dev, t_dev, p_dev;
	s_dev.upload(s);
	r_dev.upload(r);
	q_dev.upload(q);
	inv_diag_blks_dev.upload(inv_diag_blks);
	x_dev.upload(x);
	t_dev.create(x_dev.size());
	p_dev.create(x_dev.size());

	//Compute dot product on host
	float dev_dot_reduce[max_reduce_blocks];
	cudaMemcpyFromSymbol(dev_dot_reduce, device::reduce_partials_blk6x6, sizeof(float) * max_reduce_blocks, 0,
	                     cudaMemcpyDeviceToHost);
	float dev_dot = 0.0f;
	for (auto j = 0; j < num_reduce_blocks_6x6; j++)
	{
		dev_dot += dev_dot_reduce[j];
	}

	float dot_s_q = 0.0f;
	for (int j = 0; j < q.size(); j++)
	{
		dot_s_q += q[j] * s[j];
	}

	assert(std::abs((dot_s_q - dev_dot) / dev_dot) < 1e-4);

	//Download nu to compute alpha
	float nu_old_host, nu_new_host;
	cudaMemcpyFromSymbol(&nu_old_host, device::nu_old_blk6x6, sizeof(float), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&nu_new_host, device::nu_new_blk6x6, sizeof(float), 0, cudaMemcpyDeviceToHost);
	cudaSafeCall(cudaDeviceSynchronize());
	assert(std::abs(nu_new_host - nu_old_host) < 1e-7);
	const float alpha = nu_old_host / dot_s_q;

	//The value of alpha is correct
	//std::cout << "Alpha from host " << alpha << std::endl;

	//Invoke the device version function
	cudaSafeCall(cudaDeviceSynchronize());
	block6x6_pcg_kernel_1(s_dev, r_dev, q_dev, inv_diag_blks_dev, x_dev, t_dev, p_dev);
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());

	//Check x <- x + alpha * s
	for (auto i = 0; i < x.size(); i++) {
		x[i] += alpha * s[i];
	}
	std::vector<float> h_x_dev;
	x_dev.download(h_x_dev);
	assert(s.size() == x.size());
	auto max_relative_err = maxRelativeError(h_x_dev, x);
	if(max_relative_err > 1e-5) {
        std::cout << "Max relative err for x <- x + alpha s is " << max_relative_err << std::endl;
    }

	//Check t <- r - alpha * q;
	t.resize(s.size());
	std::vector<float> h_t_dev;
	t_dev.download(h_t_dev);
	for(auto j = 0;j < t.size(); j++) {
		t[j] = r[j] - alpha * q[j];
		assert(std::abs(t[j] - h_t_dev[j]) < 1e-4);
	}

	//Check p <- M_inv*t;
	std::vector<float> h_p_dev;
	p_dev.download(h_p_dev);
	p.resize(x.size());
	for (auto row = 0; row < t.size(); row++) {
		int blk_idx = row / 6;
		int inblk_offset = row % 6;
		int diag_offset = 36 * blk_idx;
		int diag_start_idx = diag_offset + 6 * inblk_offset;
		float p_row = 0.0f;
		for (auto j = 0; j < 6; j++) {
			p_row += inv_diag_blks[diag_start_idx + j] * t[6 * blk_idx + j];
		}
		p[row] = p_row;
	}
    max_relative_err = maxRelativeError(h_p_dev, p, 1e-5);
	if(max_relative_err > 1e-5) {
        std::cout << "Relative error for p <- Minv t " << max_relative_err << std::endl;
    }

	//Check for nu_new <- dot(t, p)
	float dot_t_p = 0.0f;
	for(auto j = 0; j < p.size(); j++)
	{
		//dot_t_p += h_t_dev[j] * p[j];
		dot_t_p += t[j] * p[j];
	}

	//Download the result to host
	cudaMemcpyFromSymbol(dev_dot_reduce, device::reduce_partials_blk6x6, sizeof(float) * max_reduce_blocks, 0, cudaMemcpyDeviceToHost);
	dev_dot = 0.0f;
	for (auto j = 0; j < num_reduce_blocks_6x6; j++) {
		dev_dot += dev_dot_reduce[j];
	}
	
	//Compare it
	assert(std::abs((dev_dot - dot_t_p) / dot_t_p) < 1e-4);
}


void surfelwarp::checkBlock6x6Kernel_2(
        const std::vector<float> &p,
        std::vector<float> &s
) {
    //Prepare for device input
    DeviceArray<float> p_dev, s_dev;
    assert(s.size() == p.size());
    p_dev.upload(p);
    s_dev.upload(s);

    //Compute the beta at host
    float parital_reduce[max_reduce_blocks];
    float nu_old_host;
    cudaMemcpyFromSymbol(&nu_old_host, device::nu_old_blk6x6, sizeof(float), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(parital_reduce, device::reduce_partials_blk6x6,
                         sizeof(float) * max_reduce_blocks, 0, cudaMemcpyDeviceToHost);
    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());
    float nu_new_host = 0.0f;
    for(auto j = 0; j < num_reduce_blocks_6x6; j++) {
        nu_new_host += parital_reduce[j];
    }
    float beta = nu_new_host / nu_old_host;

    //Debug code, seems correct
    //std::cout << "Beta on host " << beta << std::endl;

    //Invoke the kernel
    block6x6_pcg_kernel_2(p_dev, s_dev);

    //Download the nu_new from device
    float nu_new_device;
    cudaMemcpyFromSymbol(&nu_new_device, device::nu_new_blk6x6, sizeof(float), 0, cudaMemcpyDeviceToHost);
    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());

    //Check that value: seems correct
    assert(std::abs((nu_new_host - nu_new_device) / nu_new_host) < 1e-4);

    //Check s <- p + beta s: seems correct
    std::vector<float> h_s_dev;
    s_dev.download(h_s_dev);
    for (auto i = 0; i < h_s_dev.size(); ++i) {
        s[i] = beta * s[i] + p[i];
    }
    auto relative_err = maxRelativeError(s, h_s_dev);
    if(relative_err > 1e-4) {
        std::cout << "Max relative error in s <- p + beta s " << relative_err << std::endl;
    }
}



