#pragma once
#include "common/common_types.h"
#include "common/surfel_types.h"

namespace surfelwarp
{
	void markValidDepthPixel(
		cudaTextureObject_t depth_img,
		const unsigned rows, const unsigned cols,
		DeviceArray<char>& valid_indicator,
		cudaStream_t stream = 0
	);
	
	void collectDepthSurfel(
		cudaTextureObject_t vertex_confid_map,
		cudaTextureObject_t normal_radius_map,
		cudaTextureObject_t color_time_map,
		const DeviceArray<int>& selected_array,
		const unsigned rows, const unsigned cols,
		DeviceArray<DepthSurfel>& valid_depth_surfel,
		cudaStream_t stream = 0
	);
}