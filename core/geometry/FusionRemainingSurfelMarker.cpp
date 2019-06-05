//
// Created by wei on 5/3/18.
//

#include "common/logging.h"
#include "common/ConfigParser.h"
#include "common/Constants.h"
#include "core/geometry/FusionRemainingSurfelMarker.h"


surfelwarp::FusionRemainingSurfelMarker::FusionRemainingSurfelMarker() {
	memset(&m_fusion_maps, 0, sizeof(m_fusion_maps));
	memset(&m_live_geometry, 0, sizeof(m_live_geometry));

	m_world2camera = mat34::identity();
	m_current_time = 0;

	const auto& config = ConfigParser::Instance();
	m_intrinsic = config.rgb_intrinsic_clip();
	//m_image_rows = config.clip_image_rows();
	//m_image_cols = config.clip_image_cols();
	
	//The buffer for prefixsum
	m_remaining_indicator_prefixsum.AllocateBuffer(Constants::kMaxNumSurfels);
}


void surfelwarp::FusionRemainingSurfelMarker::SetInputs(
	const Renderer::FusionMaps& maps, 
	const SurfelGeometry::SurfelFusionInput& geometry, 
	float current_time, 
	const mat34& world2camera,
	const DeviceArraySlice<unsigned>& remaining_surfel_indicator
) {
	m_fusion_maps.vertex_confid_map = maps.warp_vertex_map;
	m_fusion_maps.normal_radius_map = maps.warp_normal_map;
	m_fusion_maps.index_map = maps.index_map;
	m_fusion_maps.color_time_map = maps.color_time_map;

	m_live_geometry.vertex_confid = geometry.live_vertex_confid.ArrayView();
	m_live_geometry.normal_radius = geometry.live_normal_radius.ArrayView();
	m_live_geometry.color_time = geometry.color_time.ArrayView();

	m_world2camera = world2camera;
	m_current_time = current_time;

	m_remaining_surfel_indicator = remaining_surfel_indicator;
	
	//Do a sanity check?
	SURFELWARP_CHECK_EQ(remaining_surfel_indicator.Size(), m_live_geometry.vertex_confid.Size());
	SURFELWARP_CHECK_EQ(remaining_surfel_indicator.Size(), m_live_geometry.normal_radius.Size());
	SURFELWARP_CHECK_EQ(remaining_surfel_indicator.Size(), m_live_geometry.color_time.Size());
}


