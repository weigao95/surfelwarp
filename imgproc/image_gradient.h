//
// Created by wei on 3/15/18.
//

#pragma once
#include "common/common_types.h"

namespace surfelwarp {
	
	
	/**
	 * \brief Compute the image gradient of filterd foreground mask and density map
	 * \param filter_foreground_mask float1 texture
	 * \param density_map float1 texture
	 * \param foreground_mask_gradient_map float2 texture
	 * \param density_gradient_map float2 texture
	 * \param rows The size of all maps
	 * \param cols  
	 */
	void computeDensityForegroundMaskGradient(
		cudaTextureObject_t filter_foreground_mask,
		cudaTextureObject_t density_map,
		unsigned rows, unsigned cols,
		cudaSurfaceObject_t foreground_mask_gradient_map,
		cudaSurfaceObject_t density_gradient_map,
		cudaStream_t stream = 0
	);

}