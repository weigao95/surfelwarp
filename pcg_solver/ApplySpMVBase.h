//
// Created by wei on 4/21/18.
//

#pragma once
#include "common/ArraySlice.h"
#include "common/macro_utils.h"
#include <memory>

namespace surfelwarp {
	
	template<int BlockDim>
	class ApplySpMVBase {
	public:
		using Ptr = std::shared_ptr<ApplySpMVBase>;
		ApplySpMVBase() = default;
		virtual ~ApplySpMVBase() = default;
		SURFELWARP_NO_COPY_ASSIGN(ApplySpMVBase);
		SURFELWARP_DEFAULT_MOVE(ApplySpMVBase);
		
		//The matrix size for this apply spmv
		virtual size_t MatrixSize() const = 0;

		//The application interface
		virtual void ApplySpMV(DeviceArrayView<float> x, DeviceArraySlice<float> spmv_x, cudaStream_t stream = 0) {
			LOG(FATAL) << "The sparse matrix vector produce is not implemented";
		}
		virtual void ApplySpMVTextured(
			cudaTextureObject_t x,
			DeviceArraySlice<float> spmv_x, 
			cudaStream_t stream = 0
		) { LOG(FATAL) << "The textured sparse matrix-vector product is not implemented"; }

		//residual <- b - Ax
		virtual void InitResidual(
			DeviceArrayView<float> x_init,
			DeviceArrayView<float> b, 
			DeviceArraySlice<float> residual, 
			cudaStream_t stream = 0
		) { LOG(FATAL) << "The init resiudal computation is not implemented"; }
		virtual void InitResidualTextured(
			cudaTextureObject_t x_init,
			DeviceArrayView<float> b, 
			DeviceArraySlice<float> residual, 
			cudaStream_t stream = 0
		) { LOG(FATAL) << "The init resiudal computation is not implemented"; }
		
		//The debug method
		static void CompareApplySpMV(typename ApplySpMVBase<BlockDim>::Ptr applier_0, typename ApplySpMVBase<BlockDim>::Ptr applier_1);
	};
}


#include "pcg_solver/ApplySpMVBase.hpp"