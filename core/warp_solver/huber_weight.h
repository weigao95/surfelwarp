#pragma once
#include <vector_types.h>
#include "core/warp_solver/solver_constants.h"

namespace surfelwarp {
	

	//The huber wight used in reweigted least square optimization, currently
	//this weight is only applied to sparse feature term as there might be outliers.
	__host__ __device__ __forceinline__ float compute_huber_weight(
		float residual,
		float residual_cutoff
	) {
		const float residual_abs = fabsf(residual);
		if(residual_abs < residual_cutoff) {
			return 1.0f;
		} 
		else {
			return residual_cutoff / residual_abs;
		}
	}


	__host__ __device__ __forceinline__ float compute_feature_huber_weight(float residual) {
		return compute_huber_weight(residual, 4e-2f);
	}

} // namespace surfelwarp