#pragma once

#include "common/ArrayView.h"


namespace surfelwarp {
	
	//The struct as the input to compactor, should be
	//used for both the regular fusion and reinitialization
	struct AppendedObservationSurfelKNN {
		//The binary indicator for the validity of each surfel
		DeviceArrayView<unsigned> validity_indicator;
		const unsigned* validity_indicator_prefixsum;
		const float4* surfel_vertex_confid;
		const float4* surfel_normal_radius;
		const float4* surfel_color_time;
		const ushort4* surfel_knn;
		const float4* surfel_knn_weight;
	};

	//The input to compactor for reinit
	struct ReinitAppendedObservationSurfel {
		DeviceArrayView<unsigned> validity_indicator;
		const unsigned* validity_indicator_prefixsum;
		cudaTextureObject_t depth_vertex_confid_map;
		cudaTextureObject_t depth_normal_radius_map;
		cudaTextureObject_t observation_color_time_map;
	};


	struct RemainingLiveSurfel {
		//The binary indicator for whether surfel i should remain
		DeviceArrayView<unsigned> remaining_indicator;
		const unsigned* remaining_indicator_prefixsum;
		const float4* live_vertex_confid;
		const float4* live_normal_radius;
		const float4* color_time;
	};

	struct RemainingSurfelKNN {
		const ushort4* surfel_knn;
		const float4* surfel_knn_weight;
	};

	struct RemainingLiveSurfelKNN {
		RemainingLiveSurfel live_geometry;
		RemainingSurfelKNN remaining_knn;
	};
}

