#pragma once

#include "imgproc/correspondence/gpc_common.h"

namespace surfelwarp {
	
	//The declare of feature build
	template<int PatchHalfSize=10>
	__device__ __forceinline__
	void buildDCTPatchFeature(
		cudaTextureObject_t normalized_rgb, int center_x, int center_y,
		GPCPatchFeature<18>& feature
	);
}

#if defined(__CUDACC__)
#include "imgproc/correspondence/gpc_feature.cuh"
#endif