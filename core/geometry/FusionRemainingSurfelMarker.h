//
// Created by wei on 5/3/18.
//

#pragma once

#include "common/ArrayView.h"
#include "common/ArraySlice.h"
#include "common/macro_utils.h"
#include "common/algorithm_types.h"
#include "math/device_mat.h"
#include "core/render/Renderer.h"
#include "core/geometry/fusion_types.h"


namespace surfelwarp {
	
	
	class FusionRemainingSurfelMarker {
	private:
		//The rendered fusion maps
		struct {
			cudaTextureObject_t vertex_confid_map;
			cudaTextureObject_t normal_radius_map;
			cudaTextureObject_t index_map;
			cudaTextureObject_t color_time_map;
		} m_fusion_maps;
		
		//The geometry model input
		struct {
			DeviceArrayView<float4> vertex_confid;
			DeviceArrayView<float4> normal_radius;
			DeviceArrayView<float4> color_time;
		} m_live_geometry;

		//The remainin surfel indicator from the fuser
		DeviceArraySlice<unsigned> m_remaining_surfel_indicator;

		//the camera and time information
		mat34 m_world2camera;
		float m_current_time;

		//The global information
		Intrinsic m_intrinsic;

	public:
		using Ptr = std::shared_ptr<FusionRemainingSurfelMarker>;
		FusionRemainingSurfelMarker();
		~FusionRemainingSurfelMarker() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(FusionRemainingSurfelMarker);

		void SetInputs(
			const Renderer::FusionMaps& maps,
			const SurfelGeometry::SurfelFusionInput& geometry,
			float current_time,
			const mat34& world2camera,
			const DeviceArraySlice<unsigned>& remaining_surfel_indicator
		);
	
		//The processing interface
		void UpdateRemainingSurfelIndicator(cudaStream_t stream = 0);
		DeviceArrayView<unsigned> GetRemainingSurfelIndicator() const { return m_remaining_surfel_indicator.ArrayView(); }
		
	private:
		PrefixSum m_remaining_indicator_prefixsum;
	public:
		void RemainingSurfelIndicatorPrefixSum(cudaStream_t stream = 0);
		DeviceArrayView<unsigned> GetRemainingSurfelIndicatorPrefixsum() const;
	};
	
	
}
