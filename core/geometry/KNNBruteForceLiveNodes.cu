#include "common/Constants.h"
#include "core/geometry/KNNBruteForceLiveNodes.h"
#include "core/geometry/brute_foce_knn.cuh"
#include "math/vector_ops.hpp"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
		
	__device__ __constant__ float4 live_node_coordinates[d_max_num_nodes];
	
	/* This kernel do skinning of vertex given the node
	 * and vertex coordinate, vertex.w can not be used
	 */
	__global__ void skinningLiveVertexBruteForceKernel(
		const DeviceArrayView<float4> vertices,
		const int node_num,
		ushort4* nearest_neighbours,
		float4* knn_weight
	) {
		//The query points and its nearest neighbour
		const int vertex_idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (vertex_idx >= vertices.Size()) return;
		const float4 vertex = vertices[vertex_idx];

		//Keep priority queue using heap
		//float d0 = 1e6f, d1 = 1e6f, d2 = 1e6f, d3 = 1e6f;
		//unsigned short i0 = 0, i1 = 0, i2 = 0, i3 = 0;
		float4 distance = make_float4(1e6f, 1e6f, 1e6f, 1e6f);
		ushort4 node_idx = make_ushort4(0, 0, 0, 0);

		//Burte force
		bruteForceSearch4Padded(vertex, live_node_coordinates, node_num, distance, node_idx);
		//bruteForceSearch4Padded(vertex, live_node_coordinates, node_num, d0, d1, d2, d3, i0, i1, i2, i3);

		//Debug check
		//if(node_idx.x != i0 || node_idx.y != i1 || node_idx.z != i2 || node_idx.w != i3) {
		//	printf("Skinning wrong\n");
		//}

		 //The forth part of vertex might be confidence
		const float3 v = make_float3(vertex.x, vertex.y, vertex.z);

		//Compute the knn weight given knn
		const float4& node0_v4 = live_node_coordinates[node_idx.x];
		const float3 node0_v = make_float3(node0_v4.x, node0_v4.y, node0_v4.z);
		const float vn_dist0 = squared_norm(v - node0_v);

		const float4& node1_v4 = live_node_coordinates[node_idx.y];
		const float3 node1_v = make_float3(node1_v4.x, node1_v4.y, node1_v4.z);
		const float vn_dist1 = squared_norm(v - node1_v);

		const float4& node2_v4 = live_node_coordinates[node_idx.z];
		const float3 node2_v = make_float3(node2_v4.x, node2_v4.y, node2_v4.z);
		const float vn_dist2 = squared_norm(v - node2_v);

		const float4& node3_v4 = live_node_coordinates[node_idx.w];
		const float3 node3_v = make_float3(node3_v4.x, node3_v4.y, node3_v4.z);
		const float vn_dist3 = squared_norm(v - node3_v);

		// Compute the weight of this node
		float4 weight;
		weight.x = __expf(-vn_dist0 / (2 * d_node_radius_square));
		weight.y = __expf(-vn_dist1 / (2 * d_node_radius_square));
		weight.z = __expf(-vn_dist2 / (2 * d_node_radius_square));
		weight.w = __expf(-vn_dist3 / (2 * d_node_radius_square));

#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION)
		const float weight_sum = weight.x + weight.y + weight.z + weight.w;
		const float inv_weight_sum = 1.0f / weight_sum;
		weight.x *= inv_weight_sum;
		weight.y *= inv_weight_sum;
		weight.z *= inv_weight_sum;
		weight.w *= inv_weight_sum;
#endif

		//Write the nearest neighbour to storage
		//nearest_neighbours[vertex_idx] = make_ushort4(i0, i1, i2, i3);
		nearest_neighbours[vertex_idx] = node_idx;
		knn_weight[vertex_idx] = weight;
	}//End of kernel

} // namespace device
} // namespace surfelwarp

/* The setup and allocation functions, all default
 */
surfelwarp::KNNBruteForceLiveNodes::KNNBruteForceLiveNodes() : m_num_nodes(0)
{
	//Allocate clear memory
	m_invalid_nodes.create(Constants::kMaxNumNodes);
	
	//The other part of the constant memory should be filled with invalid points
	std::vector<float4> h_invalid_nodes;
	h_invalid_nodes.resize(Constants::kMaxNumNodes);
	float* begin = (float*)h_invalid_nodes.data();
	float* end = begin + 4 * Constants::kMaxNumNodes;
	std::fill(begin, end, 1e6f);
	m_invalid_nodes.upload(h_invalid_nodes);
	
	//Fill the constant memory with invalid values
	clearConstantPoints();
}

surfelwarp::KNNBruteForceLiveNodes::Ptr surfelwarp::KNNBruteForceLiveNodes::Instance() {
	static KNNBruteForceLiveNodes::Ptr instance = nullptr;
	if(instance == nullptr) {
		instance.reset(new KNNBruteForceLiveNodes());
	}
	return instance;
}

surfelwarp::KNNBruteForceLiveNodes::~KNNBruteForceLiveNodes() {
	m_invalid_nodes.release();
}

void surfelwarp::KNNBruteForceLiveNodes::AllocateBuffer(unsigned max_num_points) {}

void surfelwarp::KNNBruteForceLiveNodes::ReleaseBuffer() {}


/* Clear the constant node coordinates
 */
void surfelwarp::KNNBruteForceLiveNodes::clearConstantPoints(cudaStream_t stream)
{
	cudaSafeCall(cudaMemcpyToSymbolAsync(
		device::live_node_coordinates,
		m_invalid_nodes.ptr(),
		sizeof(float4) * Constants::kMaxNumNodes,
		0, // no offset
		cudaMemcpyDeviceToDevice,
		stream
	));
}

/* Build index copy the nodes into const memory
 */
void surfelwarp::KNNBruteForceLiveNodes::BuildIndex(const DeviceArrayView<float4>& nodes, cudaStream_t stream)
{
	//If the new nodes is more than previous nodes
	if(nodes.Size() >= m_num_nodes) {
		replaceWithMorePoints(nodes, stream);
		return;
	}
	
	//First clear the buffer
	clearConstantPoints(stream);
	
	//Copy the value to device
	cudaSafeCall(cudaMemcpyToSymbolAsync(
		device::live_node_coordinates, 
		nodes.RawPtr(),
		nodes.Size() * sizeof(float4), 
		0, // no offset
		cudaMemcpyDeviceToDevice, 
		stream
	));
	
	//Update size
	m_num_nodes = nodes.Size();
}



void surfelwarp::KNNBruteForceLiveNodes::replaceWithMorePoints(const DeviceArrayView<float4>& nodes, cudaStream_t stream)
{
	SURFELWARP_CHECK_GE(nodes.Size(), m_num_nodes) << "Please use BuildIndex() instead!";
	cudaSafeCall(cudaMemcpyToSymbolAsync(
		device::live_node_coordinates,
		nodes.RawPtr(),
		nodes.Size() * sizeof(float4),
		0, // no offset
		cudaMemcpyDeviceToDevice,
		stream
	));
	m_num_nodes = nodes.Size();
}

void surfelwarp::KNNBruteForceLiveNodes::Skinning(
	const DeviceArrayView<float4>& vertex,
	DeviceArraySlice<ushort4> knn,
	DeviceArraySlice<float4> knn_weight,
	cudaStream_t stream
) {
	dim3 blk(256);
	dim3 grid(divUp(vertex.Size(), blk.x));
	device::skinningLiveVertexBruteForceKernel<<<grid, blk, 0, stream>>>(
		vertex,
		m_num_nodes,
		knn,
		knn_weight
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::KNNBruteForceLiveNodes::Skinning(
	const DeviceArrayView<float4>& vertex, const DeviceArrayView<float4>& node,
	DeviceArraySlice<ushort4> vertex_knn, DeviceArraySlice<ushort4> node_knn,
	DeviceArraySlice<float4> vertex_knn_weight, DeviceArraySlice<float4> node_knn_weight,
	cudaStream_t stream
) {
	LOG(FATAL) << "The live vertex knn searcher should not perform skinning on the nodes";
}