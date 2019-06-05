//
// Created by wei on 5/10/18.
//

#include "common/Constants.h"
#include "core/geometry/WarpFieldInitializer.h"
#include "core/geometry/WarpFieldUpdater.h"
#include "core/geometry/VoxelSubsamplerSorting.h"

surfelwarp::WarpFieldInitializer::WarpFieldInitializer() {
	m_vertex_subsampler = std::make_shared<VoxelSubsamplerSorting>();
	m_vertex_subsampler->AllocateBuffer(Constants::kMaxNumSurfels);
	m_node_candidate.AllocateBuffer(Constants::kMaxSubsampleFrom * Constants::kMaxNumNodes);
}


surfelwarp::WarpFieldInitializer::~WarpFieldInitializer() {
	m_vertex_subsampler->ReleaseBuffer();
}

void surfelwarp::WarpFieldInitializer::InitializeReferenceNodeAndSE3FromVertex(
	const DeviceArrayView<float4>& reference_vertex,
	WarpField::Ptr warp_field,
	cudaStream_t stream
) {
	//First subsampling
	performVertexSubsamplingSync(reference_vertex, stream);
	
	//Next select from candidate
	const auto& h_candidates = m_node_candidate.HostArray();
	WarpFieldUpdater::InitializeReferenceNodesAndSE3FromCandidates(*warp_field, h_candidates, stream);
}

void surfelwarp::WarpFieldInitializer::performVertexSubsamplingSync(
	const DeviceArrayView<float4>& reference_vertex,
	cudaStream_t stream
) {
	//The voxel size
	const auto subsample_voxel = 0.7f * Constants::kNodeRadius;
	
	//Perform subsampling
	auto& node_candidates = m_node_candidate;
	m_vertex_subsampler->PerformSubsample(reference_vertex, node_candidates, subsample_voxel, stream);
}