//
// Created by wei on 5/2/18.
//

#include "core/geometry/fusion_types.h"
#include "core/geometry/LiveGeometryUpdater.h"
#include "core/geometry/SurfelFusionHandler.h"
#include "core/geometry/SurfelNodeDeformer.h"
#include "core/geometry/KNNSearch.h"
#include "LiveGeometryUpdater.h"


surfelwarp::LiveGeometryUpdater::LiveGeometryUpdater(surfelwarp::SurfelGeometry::Ptr *surfel_geometry) {
	m_surfel_geometry[0] = surfel_geometry[0];
	m_surfel_geometry[1] = surfel_geometry[1];
	m_updated_idx = 0;
	
	//The buffer allocation mehtods
	m_surfel_fusion_handler = std::make_shared<SurfelFusionHandler>();
	m_fusion_remaining_surfel_marker = std::make_shared<FusionRemainingSurfelMarker>();
	m_appended_surfel_processor = std::make_shared<AppendSurfelProcessor>();
	m_surfel_compactor = std::make_shared<DoubleBufferCompactor>();
	
	//The stream
	initFusionStream();
}


surfelwarp::LiveGeometryUpdater::~LiveGeometryUpdater() {
	releaseFusionStream();
}


void surfelwarp::LiveGeometryUpdater::SetInputs(
	const Renderer::FusionMaps& maps,
	const CameraObservation& observation,
	const WarpField::LiveGeometryUpdaterInput& warpfield_input,
	const KNNSearch::Ptr& live_node_skinner,
	int updated_idx,
	float current_time,
	const mat34& world2camera
) {
	m_fusion_maps = maps;
	m_observation = observation;
	m_warpfield_input = warpfield_input;
	m_live_node_skinner = live_node_skinner;
	m_updated_idx = updated_idx % 2;
	m_current_time = current_time;
	m_world2camera = world2camera;
}

void surfelwarp::LiveGeometryUpdater::TestFusion() {
	const auto num_surfels = m_surfel_geometry[m_updated_idx]->NumValidSurfels();
	m_surfel_fusion_handler->ZeroInitializeRemainingIndicator(num_surfels);
	FuseCameraObservationSync();
	MarkRemainingSurfels();
	ProcessAppendedSurfels();
	
	//Do compaction
	unsigned num_remaining_surfel, num_appended_surfel;
	CompactSurfelToAnotherBufferSync(num_remaining_surfel, num_appended_surfel);
	
	//Do some checking on the compacted geometry
	//TestCompactionKNNFirstIter(num_remaining_surfel, num_appended_surfel);
}

void surfelwarp::LiveGeometryUpdater::ProcessFusionSerial(
	unsigned &num_remaining_surfel,
	unsigned &num_appended_surfel,
	cudaStream_t stream
) {
	const auto num_surfels = m_surfel_geometry[m_updated_idx]->NumValidSurfels();
	m_surfel_fusion_handler->ZeroInitializeRemainingIndicator(num_surfels, stream);
	FuseCameraObservationSync(stream);
	MarkRemainingSurfels(stream);
	ProcessAppendedSurfels(stream);
	CompactSurfelToAnotherBufferSync(num_remaining_surfel, num_appended_surfel, stream);
}

/* The map to perform surfel fusion
 */
void surfelwarp::LiveGeometryUpdater::FuseCameraObservationSync(cudaStream_t stream) {
	m_surfel_fusion_handler->SetInputs(
		m_fusion_maps,
		m_observation,
		m_surfel_geometry[m_updated_idx]->SurfelFusionAccess(),
		m_current_time,
		m_world2camera,
		true // Use atomic append as there are mininal appended surfels
	);
	
	//Do it
	m_surfel_fusion_handler->ProcessFusion(stream);
	m_surfel_fusion_handler->BuildCandidateAppendedPixelsSync(); //This requires sync
}


/* The buffer and method to clear the existing surfel based on knn
 */
void surfelwarp::LiveGeometryUpdater::MarkRemainingSurfels(cudaStream_t stream) {
	auto remaining_surfel_indicator = m_surfel_fusion_handler->GetFusionIndicator().remaining_surfel_indicator;
	m_fusion_remaining_surfel_marker->SetInputs(
		m_fusion_maps,
		m_surfel_geometry[m_updated_idx]->SurfelFusionAccess(),
		m_current_time,
		m_world2camera,
		remaining_surfel_indicator
	);
	
	m_fusion_remaining_surfel_marker->UpdateRemainingSurfelIndicator(stream);
	
	//Do prefixsum in another stream
	m_fusion_remaining_surfel_marker->RemainingSurfelIndicatorPrefixSum(stream);
}


surfelwarp::RemainingLiveSurfelKNN surfelwarp::LiveGeometryUpdater::GetRemainingLiveSurfelKNN() const {
	//From which geometry
	const auto geometry_from = m_surfel_geometry[m_updated_idx]->SurfelFusionAccess();
	
	RemainingLiveSurfelKNN remaining_surfel_knn;
	//The indicator part
	const auto& indicator = m_fusion_remaining_surfel_marker->GetRemainingSurfelIndicator();
	const auto& indicator_prefixsum = m_fusion_remaining_surfel_marker->GetRemainingSurfelIndicatorPrefixsum();
	SURFELWARP_CHECK(indicator.Size() == indicator_prefixsum.Size());
	remaining_surfel_knn.live_geometry.remaining_indicator = indicator;
	remaining_surfel_knn.live_geometry.remaining_indicator_prefixsum = indicator_prefixsum.RawPtr();
	
	//The geometry part
	remaining_surfel_knn.live_geometry.live_vertex_confid = geometry_from.live_vertex_confid.RawPtr();
	remaining_surfel_knn.live_geometry.live_normal_radius = geometry_from.live_normal_radius.RawPtr();
	remaining_surfel_knn.live_geometry.color_time = geometry_from.color_time.RawPtr();
	
	//The knn part
	remaining_surfel_knn.remaining_knn.surfel_knn = geometry_from.surfel_knn.RawPtr();
	remaining_surfel_knn.remaining_knn.surfel_knn_weight = geometry_from.surfel_knn_weight.RawPtr();
	return remaining_surfel_knn;
}

/* Check inconsistent skinning and collision at the appended surfels
 */
void surfelwarp::LiveGeometryUpdater::ProcessAppendedSurfels(cudaStream_t stream) {
	const auto appended_pixel = m_surfel_fusion_handler->GetFusionIndicator().appended_pixels;
	m_appended_surfel_processor->SetInputs(
		m_observation,
		m_world2camera.inverse(),
		m_warpfield_input,
		m_live_node_skinner,
		appended_pixel
	);
	
	//Do processing
	//m_appended_surfel_processor->BuildVertexForFiniteDifference(stream);
	m_appended_surfel_processor->BuildSurfelAndFiniteDiffVertex(stream);
	m_appended_surfel_processor->SkinningFiniteDifferenceVertex(stream);
	m_appended_surfel_processor->FilterCandidateSurfels(stream);
}

/* Compact surfel to another buffer
 */
void surfelwarp::LiveGeometryUpdater::CompactSurfelToAnotherBufferSync(
	unsigned& num_remaining_surfel,
	unsigned& num_appended_surfel,
	cudaStream_t stream
) {
	//Construct the remaining surfel
	RemainingLiveSurfelKNN remaining_surfel_knn = GetRemainingLiveSurfelKNN();
	
	//Construct the appended surfel
	const auto appended_surfel = m_appended_surfel_processor->GetAppendedObservationSurfel();

	//The buffer that the compactor should write to
	const auto compacted_to_idx = (m_updated_idx + 1) % 2;
	
	//Ok, seems everything is ready
	m_surfel_compactor->SetFusionInputs(remaining_surfel_knn, appended_surfel, m_surfel_geometry[compacted_to_idx]);
	m_surfel_compactor->PerformCompactionGeometryKNNSync(num_remaining_surfel, num_appended_surfel, stream);
}

void surfelwarp::LiveGeometryUpdater::TestCompactionKNNFirstIter(unsigned num_remaining_surfel, unsigned num_appended_surfel) {
	const auto compacted_to_idx = (m_updated_idx + 1) % 2;
	const auto geometry_to = m_surfel_geometry[compacted_to_idx]->SurfelFusionAccess();
	
	//Sanity check
	SURFELWARP_CHECK(geometry_to.live_vertex_confid.Size() == num_remaining_surfel + num_appended_surfel);
	
	//Check the appended surfel, they should be skinned using live nodes: seems correct
	{
		DeviceArrayView<float4> vertex = DeviceArrayView<float4>(geometry_to.live_vertex_confid.RawPtr() + num_remaining_surfel, num_appended_surfel);
		DeviceArrayView<ushort4> knn = DeviceArrayView<ushort4>(geometry_to.surfel_knn.RawPtr() + num_remaining_surfel, num_appended_surfel);
		KNNSearch::CheckKNNSearch(m_warpfield_input.live_node_coords, vertex, knn);
	}
	
	//Check the remaining surfel, should use approximate knn search
	{
		//DeviceArrayView<float4> vertex = DeviceArrayView<float4>(geometry_to.live_vertex_confid.RawPtr(), num_remaining_surfel);
		//DeviceArrayView<ushort4> knn = DeviceArrayView<ushort4>(geometry_to.surfel_knn.RawPtr(), num_remaining_surfel);
		//KNNSearch::CheckApproximateKNNSearch(m_warpfield_input.live_node_coords, vertex, knn);
	}
}


/* The method for multi-stream processing
 */
void surfelwarp::LiveGeometryUpdater::initFusionStream() {
	cudaSafeCall(cudaStreamCreate(&m_fusion_stream[0]));
	cudaSafeCall(cudaStreamCreate(&m_fusion_stream[1]));
}

void surfelwarp::LiveGeometryUpdater::releaseFusionStream() {
	cudaSafeCall(cudaStreamDestroy(m_fusion_stream[0]));
	cudaSafeCall(cudaStreamDestroy(m_fusion_stream[1]));
	m_fusion_stream[0] = 0;
	m_fusion_stream[1] = 0;
}

void surfelwarp::LiveGeometryUpdater::ProcessFusionStreamed(unsigned &num_remaining_surfel, unsigned &num_appended_surfel) {
	const auto num_surfels = m_surfel_geometry[m_updated_idx]->NumValidSurfels();
	//These two parallel
	m_surfel_fusion_handler->ZeroInitializeRemainingIndicator(num_surfels, m_fusion_stream[1]);
	FuseCameraObservationSync(m_fusion_stream[0]);
	cudaSafeCall(cudaStreamSynchronize(m_fusion_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_fusion_stream[1]));
	
	//These two parallel
	ProcessAppendedSurfels(m_fusion_stream[1]);
	MarkRemainingSurfels(m_fusion_stream[0]);
	
	//Explicit here
	cudaSafeCall(cudaStreamSynchronize(m_fusion_stream[0]));
	CompactSurfelToAnotherBufferSync(num_remaining_surfel, num_appended_surfel, m_fusion_stream[1]);
}



