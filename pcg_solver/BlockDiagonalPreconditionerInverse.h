//
// Created by wei on 4/21/18.
//

#pragma once
#include "common/macro_utils.h"
#include "common/DeviceBufferArray.h"
#include <memory>

namespace surfelwarp {
	
	template<int BlockDim>
	class BlockDiagonalPreconditionerInverse {
	public:
		using Ptr = std::shared_ptr<BlockDiagonalPreconditionerInverse>;
		BlockDiagonalPreconditionerInverse();
		BlockDiagonalPreconditionerInverse(size_t max_matrix_size);
		~BlockDiagonalPreconditionerInverse();
		SURFELWARP_NO_COPY_ASSIGN(BlockDiagonalPreconditionerInverse);
		SURFELWARP_DEFAULT_MOVE(BlockDiagonalPreconditionerInverse);

		//Allocate and release the buffer
        void AllocateBuffer(size_t max_matrix_size);
		void ReleaseBuffer();

		//The input interface
		void SetInput(DeviceArrayView<float> diagonal_blks);
		void SetInput(DeviceArray<float> diagonal_blks) {
			DeviceArrayView<float> diagonal_blks_view(diagonal_blks);
			SetInput(diagonal_blks_view);
		}

		//The processing and access interface
		void PerformDiagonalInverse(cudaStream_t stream = 0);
		DeviceArrayView<float> InversedDiagonalBlocks() const { return m_inv_diag_blks.ArrayView(); }

	private:
		//The buffer for the inverse of diagonal blocks
		DeviceBufferArray<float> m_inv_diag_blks;

		//The input to the preconditioner
		DeviceArrayView<float> m_diagonal_blks;
		size_t m_matrix_size;
	};
}

#include "pcg_solver/BlockDiagonalPreconditionerInverse.hpp"
#if defined(__CUDACC__)
#include "pcg_solver/BlockDiagonalPreconditionerInverse.cuh"
#endif
