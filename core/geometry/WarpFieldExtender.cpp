//
// Created by wei on 5/11/18.
//

#include "common/Constants.h"
#include "core/geometry/WarpFieldExtender.h"
#include "core/geometry/WarpFieldUpdater.h"

surfelwarp::WarpFieldExtender::WarpFieldExtender() {
	//As the extender might be called on the whole surfel geometry
	//allcoate the maximun amount of buffer

	m_candidate_validity_indicator.AllocateBuffer(Constants::kMaxNumSurfels);
	m_validity_indicator_prefixsum.AllocateBuffer(Constants::kMaxNumSurfels);
	m_candidate_vertex_array.AllocateBuffer(Constants::kMaxNumSurfelCandidates);
}

surfelwarp::WarpFieldExtender::~WarpFieldExtender() {
	m_candidate_validity_indicator.ReleaseBuffer();
}

void surfelwarp::WarpFieldExtender::ExtendReferenceNodesAndSE3Sync(
	const DeviceArrayView<float4>& reference_vertex,
	const DeviceArrayView<ushort4>& vertex_knn,
	WarpField::Ptr& warp_field,
	cudaStream_t stream
) {
	SURFELWARP_CHECK(reference_vertex.Size() == vertex_knn.Size());
	if(reference_vertex.Size() == 0) return;

	//First, collect the potential candidates
	const auto node_coordinates = warp_field->ReferenceNodeCoordinates().DeviceArrayReadOnly();
	labelCollectUncoveredNodeCandidate(reference_vertex, vertex_knn, node_coordinates, stream);
	syncQueryUncoveredNodeCandidateSize(stream);

	//Using the candidate to update the warp field
	if(m_candidate_vertex_array.DeviceArraySize() > 0) {
		const auto& h_candidate = m_candidate_vertex_array.HostArray();
		WarpFieldUpdater::UpdateWarpFieldFromUncoveredCandidate(*warp_field, h_candidate, stream);
	}
}


