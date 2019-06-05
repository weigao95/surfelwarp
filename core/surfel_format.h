#pragma once
#include <vector_types.h>

namespace surfelwarp {
	

	__host__ __device__ __forceinline__ float last_observed_time(const float4& color_time) {
		return color_time.z;
	}

	__host__ __device__ __forceinline__ float initialization_time(const float4& color_time) {
		return color_time.w;
	}

}