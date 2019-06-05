#include "pcg_solver/BlockPCG6x6.h"
#include "pcg_solver/block6x6_pcg_weber.h"
#include "common/common_types.h"
#include "common/common_texture_utils.h"

void surfelwarp::BlockPCG6x6::AllocateBuffer(const unsigned max_maxtrix_size)
{
	//Perform a ceiling on the input size
	const auto rectify_matrix_size = divUp(max_maxtrix_size, 6) * 6;

	//Create the buffer for arrays
	p_buffer_.create(rectify_matrix_size);
	q_buffer_.create(rectify_matrix_size);
	r_buffer_.create(rectify_matrix_size);
	s_buffer_.create(rectify_matrix_size);
	t_buffer_.create(rectify_matrix_size);
	x_buffer_.create(rectify_matrix_size);

	//Create texture object for sparse mv
	s_texture_ = create1DLinearTexture(s_buffer_);

	//Create the buffer for diag blocks
	const auto diag_blk_size = rectify_matrix_size * 6;
	inv_diag_blk_buffer_.create(diag_blk_size);
	
	//Update the size
	m_max_matrix_size = rectify_matrix_size;
}

void surfelwarp::BlockPCG6x6::ReleaseBuffer()
{
	//Release buffer for arrays
	p_buffer_.release();
	q_buffer_.release();
	r_buffer_.release();
	s_buffer_.release();
	t_buffer_.release();
	x_buffer_.release();

	//Release the texture object
	cudaSafeCall(cudaDestroyTextureObject(s_texture_));

	//Release buffer for inv-diag-blocks
	inv_diag_blk_buffer_.release();

	//Update on the size
	m_max_matrix_size = 0;
}

bool surfelwarp::BlockPCG6x6::SetSolverInput(
	const DeviceArray<float>& diag_blks, 
	const DeviceArray<float>& A_data, 
	const DeviceArray<int>& A_colptr, 
	const DeviceArray<int>& A_rowptr,
	const DeviceArray<float>& b, 
	size_t actual_size
) {
	//Modification on the size of matrix
	if(actual_size == 0) {
		actual_size = b.size();
	}
	if(actual_size % 6 != 0) {
		actual_size = divUp(actual_size, 6) * 6;
	}

	//Can not solve the matix in given size
	if(actual_size > m_max_matrix_size) {
		return false;
	}

	//Everything looks ok
	m_actual_matrix_size = actual_size;
	diag_blks_ = DeviceArray<float>((float*)diag_blks.ptr(), m_actual_matrix_size * 6);
	b_ = DeviceArray<float>((float*)b.ptr(), m_actual_matrix_size);

	//Something does not know the size
	A_data_ = DeviceArray<float>((float*)A_data.ptr(), A_data.size());
	A_rowptr_ = DeviceArray<int>((int*)A_rowptr.ptr(), A_rowptr.size());
	A_colptr_ = DeviceArray<int>((int*)A_colptr.ptr(), A_colptr.size());
	return true;
}

surfelwarp::DeviceArray<float> surfelwarp::BlockPCG6x6::Solve(
	const int max_iters, 
	cudaStream_t stream
) {
	//Correct the size for output
	DeviceArray<float> valid_x;
	
	//Lets do it
	block6x6_pcg_weber(
		diag_blks_,
		A_data_, 
		A_colptr_,
		A_rowptr_, 
		b_, 
		x_buffer_, 
		inv_diag_blk_buffer_,
		p_buffer_,
		q_buffer_, 
		r_buffer_, 
		s_buffer_,
		t_buffer_, 
		valid_x, 
		max_iters, 
		stream
	);

	return valid_x;
}

surfelwarp::DeviceArray<float> surfelwarp::BlockPCG6x6::SolveTextured(const int max_iters, cudaStream_t stream)
{
	//Correct the size for output
	DeviceArray<float> valid_x;

	//Lets do it
	block6x6_pcg_weber(
		diag_blks_,
		A_data_,
		A_colptr_,
		A_rowptr_,
		b_,
		x_buffer_,
		inv_diag_blk_buffer_,
		p_buffer_,
		q_buffer_,
		r_buffer_,
		s_buffer_,
		s_texture_,
		t_buffer_,
		valid_x,
		max_iters,
		stream
	);

	return valid_x;
}