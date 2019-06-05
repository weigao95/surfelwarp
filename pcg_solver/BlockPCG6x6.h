#pragma once

#include "common/common_types.h"
#include <memory>

namespace surfelwarp {
	
	class BlockPCG6x6 {
	public:
		using Ptr = std::shared_ptr<BlockPCG6x6>;

		//Default construct/destruct, no copy/assign/move
		explicit BlockPCG6x6() = default;
		~BlockPCG6x6() = default;
		BlockPCG6x6(const BlockPCG6x6&) = delete;
		BlockPCG6x6(BlockPCG6x6&&) = delete;
		BlockPCG6x6& operator=(const BlockPCG6x6&) = delete;
		BlockPCG6x6& operator=(BlockPCG6x6&&) = delete;

		//Allocate and release buffer explicit
		void AllocateBuffer(const unsigned max_maxtrix_size);
		void ReleaseBuffer();

		//The solver interface
		bool SetSolverInput(
			const DeviceArray<float>& diag_blks,
			const DeviceArray<float>& A_data,
			const DeviceArray<int>& A_colptr,
			const DeviceArray<int>& A_rowptr,
			const DeviceArray<float>& b,
			size_t actual_size = 0
		);

		DeviceArray<float> Solve(const int max_iters = 10, cudaStream_t stream = 0);
		DeviceArray<float> SolveTextured(const int max_iters = 10, cudaStream_t stream = 0);

	private:
		//The buffer maintained inside this class
		size_t m_max_matrix_size;
		DeviceArray<float> p_buffer_, q_buffer_, r_buffer_, s_buffer_, t_buffer_;
		DeviceArray<float> inv_diag_blk_buffer_, x_buffer_;
		cudaTextureObject_t s_texture_;

		//The buffer from setInput method
		size_t m_actual_matrix_size;
		DeviceArray<float> diag_blks_, A_data_, b_;
		DeviceArray<int> A_colptr_, A_rowptr_;
	};

}