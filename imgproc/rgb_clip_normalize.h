//
// Created by wei on 2/21/18.
//

#pragma once

#include "common/common_types.h"

namespace surfelwarp {

    void clipNormalizeRGBImage(
		const DeviceArray<uchar3> raw_rgb_img,
		const unsigned clip_rows, const unsigned clip_cols,
		cudaSurfaceObject_t clip_rgb_img,
		cudaStream_t stream = 0
	);

	void clipNormalizeRGBImage(
		const DeviceArray<uchar3>& raw_rgb_img,
		unsigned clip_rows, unsigned clip_cols,
		cudaSurfaceObject_t clip_rgb_img,
		cudaSurfaceObject_t density_map,
		cudaStream_t stream = 0
	);
	
	void filterDensityMap(
		cudaTextureObject_t density_map,
		cudaSurfaceObject_t filter_density_map,
		unsigned rows, unsigned cols,
		cudaStream_t stream = 0
	);
}