//
// Created by wei on 5/4/18.
//

#include "common/Constants.h"
#include "core/geometry/AppendSurfelProcessor.h"


surfelwarp::AppendSurfelProcessor::AppendSurfelProcessor() {
	memset(&m_observation, 0, sizeof(m_observation));
	
	//Init the warp field input
	m_warpfield_input.live_node_coords = DeviceArrayView<float4>();
	m_warpfield_input.reference_node_coords = DeviceArrayView<float4>();
	m_warpfield_input.node_se3 = DeviceArrayView<DualQuaternion>();
	m_live_node_skinner = nullptr;
	
	//The buffer for surfel and finite difference vertex
	m_surfel_vertex_confid.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
	m_surfel_normal_radius.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
	m_surfel_color_time.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
	m_candidate_vertex_finite_diff.AllocateBuffer(Constants::kMaxNumSurfelCandidates * (kNumFiniteDiffVertex));
	
	//The buffer and array for skinning
	m_candidate_vertex_finitediff_knn.AllocateBuffer(m_candidate_vertex_finite_diff.Capacity());
	m_candidate_vertex_finitediff_knnweight.AllocateBuffer(m_candidate_vertex_finite_diff.Capacity());
	
	//The indicator for the array
	m_candidate_surfel_validity_indicator.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
	m_surfel_knn.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
	m_surfel_knn_weight.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
	m_candidate_surfel_validity_prefixsum.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
}

surfelwarp::AppendSurfelProcessor::~AppendSurfelProcessor() {
	m_surfel_vertex_confid.ReleaseBuffer();
	m_surfel_normal_radius.ReleaseBuffer();
	m_surfel_color_time.ReleaseBuffer();
	m_candidate_vertex_finite_diff.ReleaseBuffer();
	
	m_candidate_vertex_finitediff_knn.ReleaseBuffer();
	m_candidate_vertex_finitediff_knnweight.ReleaseBuffer();
	
	m_candidate_surfel_validity_indicator.ReleaseBuffer();
	m_surfel_knn.ReleaseBuffer();
	m_surfel_knn_weight.ReleaseBuffer();
}

void surfelwarp::AppendSurfelProcessor::SetInputs(
	const CameraObservation &observation,
	const mat34& camera2world,
	const WarpField::LiveGeometryUpdaterInput &warpfield_input,
	const KNNSearch::Ptr& live_node_skinner,
	const DeviceArrayView<ushort2>& pixel_coordinate
) {
	m_observation.vertex_confid_map = observation.vertex_config_map;
	m_observation.normal_radius_map = observation.normal_radius_map;
	m_observation.color_time_map = observation.color_time_map;
	m_camera2world = camera2world;
	
	m_warpfield_input = warpfield_input;
	m_live_node_skinner = live_node_skinner;
	
	m_surfel_candidate_pixel = pixel_coordinate;
}

surfelwarp::AppendedObservationSurfelKNN surfelwarp::AppendSurfelProcessor::GetAppendedObservationSurfel() const {
	AppendedObservationSurfelKNN observation_surfel_knn;
	observation_surfel_knn.validity_indicator = m_candidate_surfel_validity_indicator.ArrayView();
	observation_surfel_knn.validity_indicator_prefixsum = m_candidate_surfel_validity_prefixsum.valid_prefixsum_array.ptr();
	observation_surfel_knn.surfel_vertex_confid = m_surfel_vertex_confid.Ptr();
	observation_surfel_knn.surfel_normal_radius = m_surfel_normal_radius.Ptr();
	observation_surfel_knn.surfel_color_time = m_surfel_color_time.Ptr();
	observation_surfel_knn.surfel_knn = m_surfel_knn.Ptr();
	observation_surfel_knn.surfel_knn_weight = m_surfel_knn_weight.Ptr();
	return observation_surfel_knn;
}
