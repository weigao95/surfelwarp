//
// Created by wei on 2/13/18.
//

#pragma once


#include "common/common_types.h"
#include "math/vector_ops.hpp"

#include <vector>
#include <string>
#include <Eigen/Eigen>

namespace surfelwarp {


	/**
	 * \brief The function implements Weber et al "Efficient GPU Data Structures and Methods to 
	 *        Solve Sparse Linear Systems in Dynamics Applications". This version is customized
	 *        for 6x6 blocks. The input node is assumed to be less than MACRO max_num_nodes,
	 *        so that the reduction sum during dot product can be performed using warp_scan
	 *        
	 * \param diag_blks The diagonal blocks of the JtJ matrix, assuming (6 * matrix_size) valid elements
	 * \param A_data The flattened matrix data in Bin Blocked CSR format. This array should be zero-initialized for padded element
	 * \param A_colptr The flattend colptr. NOTICE: The padded element in this array should < 0. (-1 initialized in our matrix construction)
	 * \param A_rowptr 
	 * \param b The redisual vector in A x = b
	 * \param x_buffer The memory of valid_x
	 */
	void block6x6_pcg_weber(
		const DeviceArray<float>& diag_blks,
		const DeviceArray<float>& A_data,
		const DeviceArray<int>& A_colptr,
		const DeviceArray<int>& A_rowptr,
		const DeviceArray<float>& b, // A x = b, the size of b is correct
		//Pre-allocated memory buffer
		DeviceArray<float>& x_buffer,
		DeviceArray<float>& inv_diag_blk_buffer,
		DeviceArray<float>& p_buffer,
		DeviceArray<float>& q_buffer,
		DeviceArray<float>& r_buffer,
		DeviceArray<float>& s_buffer,
		DeviceArray<float>& t_buffer,
		//The solution of A x = b, pointer to above
		DeviceArray<float>& valid_x,
		//Optional configurations
		int max_iters = 10,
		cudaStream_t stream = 0
	);

	/* The version using texture 
	 */
	void block6x6_pcg_weber(
		const DeviceArray<float>& diag_blks,
		const DeviceArray<float>& A_data,
		const DeviceArray<int>& A_colptr,
		const DeviceArray<int>& A_rowptr,
		const DeviceArray<float>& b, // A x = b, the size of b is correct
		//Pre-allocated memory buffer
		DeviceArray<float>& x_buffer,
		DeviceArray<float>& inv_diag_blk_buffer,
		DeviceArray<float>& p_buffer,
		DeviceArray<float>& q_buffer,
		DeviceArray<float>& r_buffer,
		DeviceArray<float>& s_buffer,
		cudaTextureObject_t s_texture,
		DeviceArray<float>& t_buffer,
		//The solution of A x = b, pointer to above
		DeviceArray<float>& valid_x,
		//Optional configurations
		int max_iters = 10,
		cudaStream_t stream = 0
	);

	/* Auxiliary functions for this solver, refer to 
	   Weber, et al "Efficient GPU data structures and methods to solve
	   sparse linear systems in dynamics applications"
	*/
	void block6x6_diag_inverse(const float* A, float* A_inversed, int num_matrix, cudaStream_t stream = 0);
	void block6x6_init_kernel(
		const DeviceArray<float>& b,
        const DeviceArray<float>& inv_diag_blks,
        DeviceArray<float>& r,
        DeviceArray<float>& s,
		DeviceArray<float>& x,
        cudaStream_t stream = 0
    );
    void block6x6_pcg_kernel_0(
        const DeviceArray<float>& A_data,
        const DeviceArray<int>& A_colptr,
        const DeviceArray<int>& A_rowptr,
        const DeviceArray<float>& s,
        DeviceArray<float>& q,
        cudaStream_t stream = 0
    );
	void block6x6_pcg_kernel_0(
		const DeviceArray<float>& A_data,
		const DeviceArray<int>& A_colptr,
		const DeviceArray<int>& A_rowptr,
		cudaTextureObject_t s,
		DeviceArray<float>& q,
		cudaStream_t stream = 0
	);
    void block6x6_pcg_kernel_1(
        const DeviceArray<float>& s,
        const DeviceArray<float>& r,
        const DeviceArray<float>& q,
        const DeviceArray<float>& inv_diag_blks,
        DeviceArray<float>& x,
        DeviceArray<float>& t,
        DeviceArray<float>& p,
        cudaStream_t stream = 0
    );


	/**
	* \brief nu_new <- dot(t, p); beta <- nu_new/nu_old; s <- p + beta s
	*/
	void block6x6_pcg_kernel_2(
		const DeviceArray<float>& p,
		DeviceArray<float>& s,
		cudaStream_t stream = 0
	);
}


/*
 * The check functions for this class
 */
namespace surfelwarp {
    //Load the check data
    void loadCheckData(
            std::vector<float>& A_data,
            std::vector<int>& A_rowptr,
            std::vector<int>& A_colptr,
            std::vector<float>& b,
            std::vector<float>& diag_blks
    );
    void loadCheckData(
            std::vector<float>& x,
            std::vector<float>& inv_diag_blks
    );

    //Check the matrix inverse
    void check6x6DiagBlocksInverse(
            const std::vector<float>& diag_blks,
            const std::vector<float>& inv_diag_blks
    );

    //Check the init kernel
    void checkBlock6x6Init(
            const std::vector<float>& b,
            const std::vector<float>& inv_diags,
            std::vector<float>& h_r,
            std::vector<float>& h_s
    );
    void checkBlock6x6Init(const std::vector<float>& b,
                           const std::vector<float>& inv_diags);

    //Check the kernel_0
    void checkBlock6x6Kernel_0(
            const std::vector<float>& A_data,
            const std::vector<int>& A_rowptr,
            const std::vector<int>& A_colptr,
            const std::vector<float>& s,
            //Output for later checking
            std::vector<float>& q
    );

    //Build the triplet matrix on host
    void block6x6BuildTripletVector(
		const std::vector<float>& A_data,
		const std::vector<int>& A_rowptr,
		const std::vector<int>& A_colptr,
		const int matrix_size,
		std::vector<Eigen::Triplet<float>>& tripletVec
    );
	void hostEigenSpMV(
		const std::vector<float>& A_data,
		const std::vector<int>& A_rowptr,
		const std::vector<int>& A_colptr,
		const int matrix_size,
		const std::vector<float>& x,
		std::vector<float>& spmv
	);
	void hostEigenSpMV(
		const std::vector<float>& A_data,
		const std::vector<int>& A_rowptr,
		const std::vector<int>& A_colptr,
		const int matrix_size,
		const Eigen::VectorXf& s,
		std::vector<float>& spmv
	);

    //Check the kernel 1
    void checkBlock6x6Kernel_1(
        const std::vector<float>& s,
        const std::vector<float>& r,
        const std::vector<float>& q,
        const std::vector<float>& inv_diag_blks,
        std::vector<float>& x,
        std::vector<float>& t,
        std::vector<float>& p
    );

	//CHeck the kernel 2
	void checkBlock6x6Kernel_2(const std::vector<float>& p, std::vector<float>& s);

    //Check the whole solver
    void checkBlock6x6PCGSolver(
            const std::vector<float>& diag_blks,
            const std::vector<float>& A_data,
            const std::vector<int>& A_colptr,
            const std::vector<int>& A_rowptr,
            const std::vector<float>& b,
            std::vector<float>& x
    );
}