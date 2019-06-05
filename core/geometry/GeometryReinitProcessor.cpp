#include "common/ConfigParser.h"
#include "common/Constants.h"
#include "core/geometry/GeometryReinitProcessor.h"


surfelwarp::GeometryReinitProcessor::GeometryReinitProcessor(SurfelGeometry::Ptr surfel_geometry[2]) {
	m_surfel_geometry[0] = surfel_geometry[0];
	m_surfel_geometry[1] = surfel_geometry[1];
	m_updated_idx = 0;

	//Init of other attributes
	m_surfel_fusion_handler = std::make_shared<SurfelFusionHandler>();
	m_remaining_surfel_marker = std::make_shared<ReinitRemainingSurfelMarker>();
	
	//The buffer for prefix sum
	const auto& config = ConfigParser::Instance();
	const auto image_size = config.clip_image_rows() * config.clip_image_cols();
	m_appended_indicator_prefixsum.AllocateBuffer(image_size);
	m_remaining_indicator_prefixsum.AllocateBuffer(Constants::kMaxNumSurfels);
	
	//The buffer for compactor
	m_surfel_compactor = std::make_shared<DoubleBufferCompactor>();
}

surfelwarp::GeometryReinitProcessor::~GeometryReinitProcessor()
{
}

void surfelwarp::GeometryReinitProcessor::SetInputs(
	const Renderer::FusionMaps & maps,
	const CameraObservation & observation, 
	int updated_idx, float current_time, 
	const mat34 & world2camera
) {
	m_fusion_maps = maps;
	m_observation = observation;
	m_updated_idx = updated_idx % 2;
	m_current_time = current_time;
	m_world2camera = world2camera;
}

void surfelwarp::GeometryReinitProcessor::ProcessReinitObservedOnlySerial(
	unsigned &num_remaining_surfel,
	unsigned &num_appended_surfel,
	cudaStream_t stream
) {
	const auto num_surfels = m_surfel_geometry[m_updated_idx]->NumValidSurfels();
	m_surfel_fusion_handler->ZeroInitializeRemainingIndicator(num_surfels, stream);
	FuseCameraObservationNoSync(stream);
	MarkRemainingSurfelObservedOnly(stream);
	
	//Do prefix sum
	processAppendedIndicatorPrefixsum(stream);
	processRemainingIndicatorPrefixsum(stream);
	
	//Ready for compaction
	CompactSurfelToAnotherBufferSync(num_remaining_surfel, num_appended_surfel, stream);
}


void surfelwarp::GeometryReinitProcessor::ProcessReinitNodeErrorSerial(
	unsigned & num_remaining_surfel, 
	unsigned & num_appended_surfel, 
	const NodeAlignmentError & node_error, 
	float threshold, 
	cudaStream_t stream
) {
	const auto num_surfels = m_surfel_geometry[m_updated_idx]->NumValidSurfels();
	m_surfel_fusion_handler->ZeroInitializeRemainingIndicator(num_surfels, stream);
	FuseCameraObservationNoSync(stream);
	MarkRemainingSurfelNodeError(node_error, threshold, stream);
	
	//Do prefix sum
	processAppendedIndicatorPrefixsum(stream);
	processRemainingIndicatorPrefixsum(stream);
	
	//Ready for compaction
	CompactSurfelToAnotherBufferSync(num_remaining_surfel, num_appended_surfel, stream);
}

//Fuse camera observation into surfel geometry
void surfelwarp::GeometryReinitProcessor::FuseCameraObservationNoSync(cudaStream_t stream) {
	m_surfel_fusion_handler->SetInputs(
		m_fusion_maps,
		m_observation,
		m_surfel_geometry[m_updated_idx]->SurfelFusionAccess(),
		m_current_time,
		m_world2camera,
		false // Use compaction append as there might be many appended surfels
	);
	
	//Do not use the internal candidate builder
	m_surfel_fusion_handler->ProcessFusionReinit(stream);
}


//Update the remaining indicator of the surfels
void surfelwarp::GeometryReinitProcessor::MarkRemainingSurfelObservedOnly(cudaStream_t stream) {
	//The input from fuser
	auto remaining_indicator = m_surfel_fusion_handler->GetRemainingSurfelIndicator();
	
	//hand in to marker
	m_remaining_surfel_marker->SetInputs(
		m_fusion_maps,
		m_surfel_geometry[m_updated_idx]->SurfelFusionAccess(),
		m_observation,
		m_current_time,
		m_world2camera,
		remaining_indicator
	);
	
	m_remaining_surfel_marker->MarkRemainingSurfelObservedOnly(stream);
}

void surfelwarp::GeometryReinitProcessor::MarkRemainingSurfelNodeError(
	const NodeAlignmentError & node_error, 
	float threshold, 
	cudaStream_t stream
) {
	//The input from fuser
	auto remaining_indicator = m_surfel_fusion_handler->GetRemainingSurfelIndicator();
	
	//hand in to marker
	m_remaining_surfel_marker->SetInputs(
		m_fusion_maps,
		m_surfel_geometry[m_updated_idx]->SurfelFusionAccess(),
		m_observation,
		m_current_time,
		m_world2camera,
		remaining_indicator
	);
	
	m_remaining_surfel_marker->MarkRemainingSurfelNodeError(node_error, threshold, stream);
}


//Compute the prefixsum for remaining indicator and appended indicator
void surfelwarp::GeometryReinitProcessor::processRemainingIndicatorPrefixsum(cudaStream_t stream) {
	const auto remaining_indicator = m_remaining_surfel_marker->GetRemainingSurfelIndicator();
	m_remaining_indicator_prefixsum.InclusiveSum(remaining_indicator.ArrayView(), stream);
}

void surfelwarp::GeometryReinitProcessor::processAppendedIndicatorPrefixsum(cudaStream_t stream) {
	const auto appended_indicator = m_surfel_fusion_handler->GetAppendedObservationCandidateIndicator();
	m_appended_indicator_prefixsum.InclusiveSum(appended_indicator, stream);
}

surfelwarp::RemainingLiveSurfel surfelwarp::GeometryReinitProcessor::getCompactionRemainingSurfel() const {
	RemainingLiveSurfel remaining_surfel;
	
	//The indicator after marker
	remaining_surfel.remaining_indicator = m_remaining_surfel_marker->GetRemainingSurfelIndicator().ArrayView();
	
	//Check and attach prefix sum
	const auto& prefixsum_array = m_remaining_indicator_prefixsum.valid_prefixsum_array;
	SURFELWARP_CHECK(remaining_surfel.remaining_indicator.Size() == prefixsum_array.size());
	remaining_surfel.remaining_indicator_prefixsum = prefixsum_array.ptr();
	
	//Check and attach geometry
	const auto& fusion_access = m_surfel_geometry[m_updated_idx]->SurfelFusionAccess();
	SURFELWARP_CHECK(remaining_surfel.remaining_indicator.Size() == fusion_access.live_vertex_confid.Size());
	SURFELWARP_CHECK(remaining_surfel.remaining_indicator.Size() == fusion_access.live_normal_radius.Size());
	SURFELWARP_CHECK(remaining_surfel.remaining_indicator.Size() == fusion_access.color_time.Size());
	remaining_surfel.live_vertex_confid = fusion_access.live_vertex_confid.RawPtr();
	remaining_surfel.live_normal_radius = fusion_access.live_normal_radius.RawPtr();
	remaining_surfel.color_time = fusion_access.color_time.RawPtr();
	
	//Everything is ready
	return remaining_surfel;
}

surfelwarp::ReinitAppendedObservationSurfel surfelwarp::GeometryReinitProcessor::getCompactionAppendedSurfel() const {
	ReinitAppendedObservationSurfel appended_observation;
	
	//The indicator directly from the fuser
	appended_observation.validity_indicator = m_surfel_fusion_handler->GetAppendedObservationCandidateIndicator();
	
	//Check and attach prefixsum
	const auto& prefixsum_array = m_appended_indicator_prefixsum.valid_prefixsum_array;
	SURFELWARP_CHECK(appended_observation.validity_indicator.Size() == prefixsum_array.size());
	appended_observation.validity_indicator_prefixsum = prefixsum_array.ptr();
	
	//The texture object from observation
	appended_observation.depth_vertex_confid_map = m_observation.vertex_config_map;
	appended_observation.depth_normal_radius_map = m_observation.normal_radius_map;
	appended_observation.observation_color_time_map = m_observation.color_time_map;
	
	return appended_observation;
}

void surfelwarp::GeometryReinitProcessor::CompactSurfelToAnotherBufferSync(
	unsigned &num_remaining_surfels,
	unsigned &num_appended_surfels,
	cudaStream_t stream
) {
	//The input
	const auto& remaining_surfels = getCompactionRemainingSurfel();
	const auto& appended_surfels = getCompactionAppendedSurfel();
	
	//The output
	const auto compacted_to_idx = (m_updated_idx + 1) % 2;
	
	//Do compaction
	m_surfel_compactor->SetReinitInputs(remaining_surfels, appended_surfels, m_surfel_geometry[compacted_to_idx]);
	m_surfel_compactor->PerformComapctionGeometryOnlySync(num_remaining_surfels, num_appended_surfels, m_world2camera.inverse(), stream);
}
