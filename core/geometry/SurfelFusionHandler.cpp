//
// Created by wei on 5/2/18.
//

#include "common/Constants.h"
#include "common/ConfigParser.h"
#include "common/sanity_check.h"
#include "core/geometry/SurfelFusionHandler.h"

#include <map>

surfelwarp::SurfelFusionHandler::SurfelFusionHandler() {
	memset(&m_fusion_maps, 0, sizeof(m_fusion_maps));
	memset(&m_fusion_geometry, 0, sizeof(m_fusion_geometry));
	
	//The surfel indicator is in the size of maximun surfels
	m_remaining_surfel_indicator.AllocateBuffer(Constants::kMaxNumSurfels);
	
	//The append depth indicator is always in the same size as image pixels
	const auto& config = ConfigParser::Instance();
	const auto pixel_size = config.clip_image_rows() * config.clip_image_cols();
	m_appended_depth_surfel_indicator.create(pixel_size);
	m_appended_surfel_indicator_prefixsum.AllocateBuffer(pixel_size);
	m_compacted_appended_pixel.AllocateBuffer(pixel_size);
	
	//The rows and cols of the image
	m_image_cols = config.clip_image_cols();
	m_image_rows = config.clip_image_rows();
	
	//The buffer for atomic appending
	cudaSafeCall(cudaMalloc(&m_atomic_appended_pixel_index, sizeof(unsigned)));
	m_atomic_appended_observation_pixel.AllocateBuffer(m_image_rows * m_image_cols);
}

surfelwarp::SurfelFusionHandler::~SurfelFusionHandler() {
	m_remaining_surfel_indicator.ReleaseBuffer();
	m_appended_depth_surfel_indicator.release();
	
	//The buffer for atomic appending
	cudaSafeCall(cudaFree(m_atomic_appended_pixel_index));
	m_atomic_appended_observation_pixel.ReleaseBuffer();
}

void surfelwarp::SurfelFusionHandler::SetInputs(
	const Renderer::FusionMaps& maps,
	const CameraObservation& observation,
	const SurfelGeometry::SurfelFusionInput& geometry,
	float current_time,
	const mat34& world2camera,
	bool use_atomic_append
) {
	m_fusion_maps = maps;
	m_observation = observation;
	m_fusion_geometry = geometry;
	m_current_time = current_time;
	m_world2camera = world2camera;
	m_use_atomic_append = use_atomic_append;
}


surfelwarp::SurfelFusionHandler::FusionIndicator surfelwarp::SurfelFusionHandler::GetFusionIndicator() {
	FusionIndicator indicator;
	indicator.remaining_surfel_indicator = m_remaining_surfel_indicator.ArraySlice();
	//This also depends on whether using atomic append
	if(m_use_atomic_append) indicator.appended_pixels = m_atomic_appended_observation_pixel.ArrayView();
	else indicator.appended_pixels = m_compacted_appended_pixel.ArrayView();
	return indicator;
}


void surfelwarp::SurfelFusionHandler::ZeroInitializeRemainingIndicator(unsigned num_surfels, cudaStream_t stream) {
	cudaSafeCall(cudaMemsetAsync(
		m_remaining_surfel_indicator.Ptr(),
		0, num_surfels * sizeof(unsigned), 
		stream
	));
	m_remaining_surfel_indicator.ResizeArrayOrException(num_surfels);
}

surfelwarp::DeviceArraySlice<unsigned> surfelwarp::SurfelFusionHandler::GetRemainingSurfelIndicator() {
	return m_remaining_surfel_indicator.ArraySlice();
}

//Only meaningful when using compaction append
surfelwarp::DeviceArrayView<unsigned> surfelwarp::SurfelFusionHandler::GetAppendedObservationCandidateIndicator() const {
	return DeviceArrayView<unsigned>(m_appended_depth_surfel_indicator);
}

void surfelwarp::SurfelFusionHandler::ProcessFusion(cudaStream_t stream) {
	//Debug check
	//SURFELWARP_CHECK(!containsNaN(m_fusion_geometry.live_vertex_confid.ArrayView()));
	
	//Do fusion
	if(m_use_atomic_append) processFusionAppendAtomic(stream);
	else processFusionAppendCompaction(stream);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Debug method
	//fusionStatistic(kUseAtomicAppend);
	//confidenceStatistic();
}

void surfelwarp::SurfelFusionHandler::ProcessFusionReinit(cudaStream_t stream) {
	processFusionReinit(stream);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::SurfelFusionHandler::BuildCandidateAppendedPixelsSync(cudaStream_t stream) {
	if(m_use_atomic_append) queryAtomicAppendedPixelSize(stream);
	else compactAppendedIndicator(stream);
}


/* These are debug methods
 */
void surfelwarp::SurfelFusionHandler::fusionStatistic(bool using_atomic) {
	LOG(INFO) << "The total number of surfel is " << m_remaining_surfel_indicator.ArraySize();
	LOG(INFO) << "The number of fused surfel is " << numNonZeroElement(m_remaining_surfel_indicator.ArrayView());
	if(using_atomic) {
		unsigned num_appended_surfels = 0;
		cudaSafeCall(cudaMemcpy(&num_appended_surfels, m_atomic_appended_pixel_index, sizeof(unsigned), cudaMemcpyDeviceToHost));
		LOG(INFO) << "The number of appended observation surfel is " << num_appended_surfels;
	} else {
		LOG(INFO) << "The number of appended observation surfel is " << numNonZeroElement(DeviceArrayView<unsigned>(m_appended_depth_surfel_indicator));
	}
}


void surfelwarp::SurfelFusionHandler::confidenceStatistic() {
	//Download the vertex confidence array
	std::vector<float4> h_vertex_confid;
	m_fusion_geometry.live_vertex_confid.SyncToHost(h_vertex_confid);

	std::map<unsigned, unsigned> confid2number;
	for(auto i = 0; i < h_vertex_confid.size(); i++) {
		float confidence = h_vertex_confid[i].w;
		confidence += 0.01;
		if(confidence < 1.0f) {
			LOG(INFO) << "The vertex " << i;
		}
		const auto uint_confid = unsigned(confidence);
		auto ptr = confid2number.find(uint_confid);
		if(ptr == confid2number.end()) {
			confid2number[uint_confid] = 1;
		} else {
			const auto value = ptr->second;
			confid2number[uint_confid] = value + 1;
		}
	}

	LOG(INFO) << "The confidence of surfels";
	for(auto iter = confid2number.begin(); iter != confid2number.end(); iter++) {
		LOG(INFO) << "The confidence at " << iter->first << " has " << iter->second << " surfels";
	}
	LOG(INFO) << "End of the confidence of surfels";
}



