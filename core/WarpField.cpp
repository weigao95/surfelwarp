//
// Created by wei on 3/24/18.
//

#include "common/safe_call_utils.hpp"
#include "math/DualQuaternion.hpp"
#include "core/WarpField.h"
#include "core/geometry/NodeGraphBuilderNaive.h"
#include "core/geometry/KNNSearch.h"
#include "visualization/Visualizer.h"

#include <cuda.h>

/* The initialization of warp field
 */
surfelwarp::WarpField::WarpField() {
	allocateBuffer(Constants::kMaxNumNodes);
}

surfelwarp::WarpField::~WarpField() {
	releaseBuffer();
}

void surfelwarp::WarpField::allocateBuffer(size_t max_num_nodes) {
	//The buffer on both host and device
	m_reference_node_coords.AllocateBuffer(max_num_nodes);
	m_node_se3.AllocateBuffer(max_num_nodes);
	m_node_knn.AllocateBuffer(max_num_nodes);
	m_node_knn_weight.AllocateBuffer(max_num_nodes);
	
	//The device only buffer
	m_live_node_coords.AllocateBuffer(max_num_nodes);
	m_node_graph.AllocateBuffer(max_num_nodes * Constants::kNumNodeGraphNeigbours);

	//The buffer for KNN, no-longer required
	//m_reference_node_skinning_index = KNNBruteForceReferenceNodes::Instance();
	
	//The method for build node graph
	m_node_graph_builder = std::make_shared<NodeGraphBuilderNaive>();
}

void surfelwarp::WarpField::releaseBuffer() {
	m_live_node_coords.ReleaseBuffer();
	m_node_graph.ReleaseBuffer();
}

void surfelwarp::WarpField::ResizeDeviceArrayToNodeSize(unsigned node_size) {
	m_node_knn.ResizeArrayOrException(node_size);
	m_node_knn_weight.ResizeArrayOrException(node_size);
	m_live_node_coords.ResizeArrayOrException(node_size);
}

unsigned surfelwarp::WarpField::CheckAndGetNodeSize() const {
	const auto num_nodes = m_reference_node_coords.DeviceArraySize();
	SURFELWARP_CHECK(num_nodes == m_reference_node_coords.HostArraySize());
	SURFELWARP_CHECK(num_nodes == m_node_se3.HostArraySize());
	SURFELWARP_CHECK(num_nodes == m_node_se3.DeviceArraySize());
	SURFELWARP_CHECK(num_nodes == m_node_knn.ArraySize());
	SURFELWARP_CHECK(num_nodes == m_node_knn_weight.ArraySize());
	SURFELWARP_CHECK(num_nodes == m_live_node_coords.ArraySize());
	return num_nodes;
}

/* Thin warper for the node graph builder
 */
void surfelwarp::WarpField::BuildNodeGraph(cudaStream_t stream) {
	m_node_graph_builder->BuildNodeGraph(
		m_reference_node_coords.DeviceArrayReadOnly(),
		m_node_graph,
		stream
	);
}


/* The accessing methods
 */
surfelwarp::WarpField::SolverInput surfelwarp::WarpField::SolverAccess() const {
	//Debug
	//LOG(INFO) << "Random warp field input";
	//DeviceArraySlice<DualQuaternion> node_se3_slice((DualQuaternion*)m_node_se3.DevicePtr(), m_node_se3.DeviceArraySize());
	//applyRandomSE3ToWarpField(node_se3_slice, 0.01, 0.02);
	
	SolverInput solver_input;
	solver_input.node_se3 = m_node_se3.DeviceArrayReadOnly();
	solver_input.reference_node_coords = m_reference_node_coords.DeviceArrayReadOnly();
	solver_input.node_graph = m_node_graph.ArrayReadOnly();
	return solver_input;
}


surfelwarp::WarpField::LiveGeometryUpdaterInput surfelwarp::WarpField::GeometryUpdaterAccess() const {
	LiveGeometryUpdaterInput geometry_input;
	geometry_input.live_node_coords = m_live_node_coords.ArrayView();
	geometry_input.reference_node_coords = m_reference_node_coords.DeviceArrayReadOnly();
	geometry_input.node_se3 = m_node_se3.DeviceArrayReadOnly();
	return geometry_input;
}


surfelwarp::WarpField::SkinnerInput surfelwarp::WarpField::SkinnerAccess() {
	SkinnerInput skinner_input;
	skinner_input.reference_node_coords = m_reference_node_coords.DeviceArrayReadOnly();
	skinner_input.node_knn = m_node_knn.ArraySlice();
	skinner_input.node_knn_weight = m_node_knn_weight.ArraySlice();
	return skinner_input;
}


/* The updating methods
 */
void surfelwarp::WarpField::UpdateHostDeviceNodeSE3NoSync(surfelwarp::DeviceArrayView<surfelwarp::DualQuaternion> node_se3, cudaStream_t stream) {
	SURFELWARP_CHECK(node_se3.Size() == m_node_se3.DeviceArraySize());
	cudaSafeCall(cudaMemcpyAsync(
		m_node_se3.DevicePtr(),
		node_se3.RawPtr(),
		sizeof(DualQuaternion) * node_se3.Size(),
		cudaMemcpyDeviceToDevice,
		stream
	));
	
	//Sync to host
	m_node_se3.SynchronizeToHost(stream, false);
}


/* These are debug methods
 */
surfelwarp::WarpField::LegacySolverAccess surfelwarp::WarpField::LegacySolverInput() {
	LegacySolverAccess solver_input;
	solver_input.node_se3 = DeviceArray<DualQuaternion>(m_node_se3.DevicePtr(), m_node_se3.DeviceArraySize());
	solver_input.reference_node_coords = DeviceArray<float4>(m_reference_node_coords.DevicePtr(), m_reference_node_coords.DeviceArraySize());
	solver_input.node_graph = DeviceArray<ushort2>(m_node_graph.Ptr(), m_node_graph.ArraySize());
	return solver_input;
}

void surfelwarp::WarpField::CheckNodeKNN() {
	KNNSearch::CheckKNNSearch(
		m_reference_node_coords.DeviceArrayReadOnly(),
		m_reference_node_coords.DeviceArrayReadOnly(),
		m_node_knn.ArrayView()
	);
}







