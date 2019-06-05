//
// Created by wei on 4/5/18.
//
#include "common/Constants.h"
#include "common/ConfigParser.h"
#include "core/warp_solver/SparseCorrespondenceHandler.h"

void surfelwarp::SparseCorrespondenceHandler::AllocateBuffer() {
	const auto max_features = Constants::kMaxMatchedSparseFeature;
	
	m_valid_pixel_indicator.AllocateBuffer(max_features);
	m_valid_pixel_prefixsum.AllocateBuffer(max_features);
	m_corrected_pixel_pairs.AllocateBuffer(max_features);
	
	m_valid_target_vertex.AllocateBuffer(max_features);
	m_valid_reference_vertex.AllocateBuffer(max_features);
	m_valid_vertex_knn.AllocateBuffer(max_features);
	m_valid_knn_weight.AllocateBuffer(max_features);
	m_valid_warped_vertex.AllocateBuffer(max_features);

	cudaSafeCall(cudaMallocHost((void**)(&m_correspondence_array_size), sizeof(unsigned)));
}

void surfelwarp::SparseCorrespondenceHandler::ReleaseBuffer() {
	m_valid_pixel_indicator.ReleaseBuffer();
	m_corrected_pixel_pairs.ReleaseBuffer();
	
	m_valid_target_vertex.ReleaseBuffer();
	m_valid_reference_vertex.ReleaseBuffer();
	m_valid_vertex_knn.ReleaseBuffer();
	m_valid_knn_weight.ReleaseBuffer();

	cudaSafeCall(cudaFreeHost(m_correspondence_array_size));
}


/* The main processing interface
 */
void surfelwarp::SparseCorrespondenceHandler::SetInputs(
	DeviceArrayView<DualQuaternion> node_se3,
	DeviceArrayView2D<KNNAndWeight> knn_map,
	cudaTextureObject_t depth_vertex_map,
	DeviceArrayView<ushort4> correspond_pixel_pairs,
	cudaTextureObject_t reference_vertex_map,
	cudaTextureObject_t index_map,
	const mat34& world2camera
) {
	m_observations.depth_vertex_map = depth_vertex_map;
	m_observations.correspond_pixel_pairs = correspond_pixel_pairs;

	m_geometry_maps.reference_vertex_map = reference_vertex_map;
	m_geometry_maps.index_map = index_map;
	m_geometry_maps.knn_map = knn_map;

	m_node_se3 = node_se3;
	m_camera2world = world2camera.inverse();
}


void surfelwarp::SparseCorrespondenceHandler::UpdateNodeSE3(DeviceArrayView<DualQuaternion> node_se3) {
	SURFELWARP_CHECK(m_node_se3.Size() == node_se3.Size());
	m_node_se3 = node_se3;
}


/* Build the correspondence vertex pairs
 */
void surfelwarp::SparseCorrespondenceHandler::BuildCorrespondVertexKNN(cudaStream_t stream) {
	ChooseValidPixelPairs(stream);
	CompactQueryPixelPairs(stream);
	QueryCompactedArraySize(stream); //This will sync host threads with stream
}






