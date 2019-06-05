#include "common/ConfigParser.h"
#include "core/geometry/ReinitRemainingSurfelMarker.h"


surfelwarp::ReinitRemainingSurfelMarker::ReinitRemainingSurfelMarker() {
	memset(&m_surfel_geometry, 0, sizeof(m_surfel_geometry));
	memset(&m_observation, 0, sizeof(m_observation));
	memset(&m_fusion_maps, 0, sizeof(m_fusion_maps));
	m_world2camera = mat34::identity();

	//The projection intrinsic
	const auto& config = ConfigParser::Instance();
	m_intrinsic = config.rgb_intrinsic_clip();
}

void surfelwarp::ReinitRemainingSurfelMarker::SetInputs(
	const Renderer::FusionMaps & maps, 
	const SurfelGeometry::SurfelFusionInput & geometry,
	const CameraObservation& observation,
	float current_time, 
	const mat34 & world2camera, 
	const DeviceArraySlice<unsigned>& remaining_surfel_indicator
) {
	m_fusion_maps = maps;
	m_surfel_geometry = geometry;
	m_observation = observation;
	m_world2camera = world2camera;
	m_remaining_surfel_indicator = remaining_surfel_indicator;

	//Sanity check
	SURFELWARP_CHECK_EQ(remaining_surfel_indicator.Size(), geometry.live_vertex_confid.Size());
	SURFELWARP_CHECK_EQ(remaining_surfel_indicator.Size(), geometry.live_normal_radius.Size());
	SURFELWARP_CHECK_EQ(remaining_surfel_indicator.Size(), geometry.color_time.Size());
}