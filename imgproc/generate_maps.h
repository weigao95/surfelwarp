#pragma once
#include "common/common_types.h"

namespace surfelwarp
{

	void createVertexConfigMap(
		cudaTextureObject_t depth_img,
		const unsigned rows, const unsigned cols,
		const IntrinsicInverse intrinsic_inv,
		cudaSurfaceObject_t vertex_confid_map,
		cudaStream_t stream = 0
	);

	void createNormalRadiusMap(
		cudaTextureObject_t vertex_map,
		const unsigned rows, const unsigned cols,
		cudaSurfaceObject_t normal_radius_map,
		cudaStream_t stream = 0
	);

	void createColorTimeMap(
		const DeviceArray<uchar3> raw_rgb_img,
		const unsigned clip_rows, const unsigned clip_cols,
		const float init_time,
		cudaSurfaceObject_t color_time_map,
		cudaStream_t stream = 0
	);
}