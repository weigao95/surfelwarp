//
// Created by wei on 3/20/18.
//

#pragma once

#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/surfel_types.h"

namespace surfelwarp {
	
	struct CameraObservation {
		//The raw depth image for visualization
		cudaTextureObject_t raw_depth_img;

		//The geometry member
		cudaTextureObject_t filter_depth_img;
		cudaTextureObject_t vertex_config_map;
		cudaTextureObject_t normal_radius_map;
		
		//The color member
		cudaTextureObject_t color_time_map;
		cudaTextureObject_t normalized_rgba_map;
		cudaTextureObject_t normalized_rgba_prevframe; // normalized rgba float4 texture
		cudaTextureObject_t density_map;
		//cudaTextureObject_t density_map_prevframe;
		cudaTextureObject_t density_gradient_map;
		
		//The compacted surfels
		//DeviceArrayView<DepthSurfel> valid_surfel_array;
		
		//The segmented mask
		cudaTextureObject_t foreground_mask;
		cudaTextureObject_t filter_foreground_mask;
		cudaTextureObject_t foreground_mask_gradient_map;
		
		//The sparse feature correspondence
		DeviceArrayView<ushort4> correspondence_pixel_pairs;
	};
}
