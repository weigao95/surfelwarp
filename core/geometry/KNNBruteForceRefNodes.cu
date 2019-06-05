#include "common/global_configs.h"
#include "common/Constants.h"
#include "common/common_utils.h"
#include "math/vector_ops.hpp"
#include "core/geometry/KNNBruteForceRefNodes.h"
#include "core/geometry/brute_foce_knn.cuh"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__device__ __constant__ float4 reference_node_coordinates[d_max_num_nodes];


	/* This kernel do skinning of both vertex and nodes given
	 * node coordinate and vertex coordinate, vertex.w can not be used
	 */
	__global__ void skinningVertexNodeBruteForceKernel(
		const DeviceArrayView<float4> vertex_confid_array,
		const int node_num,
		ushort4* vertex_knn_array, float4* vertex_knn_weight,
		ushort4* node_knn_array, float4* node_knn_weight
	) {
		// Outof bound: for both vertex and node knn are updated by this kernel
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= vertex_confid_array.Size() + node_num) return;

		// Load the vertex from memory
		float4 vertex;
		if (idx < vertex_confid_array.Size()) {
			const float4 vertex_confid = vertex_confid_array[idx];
			vertex = make_float4(vertex_confid.x, vertex_confid.y, vertex_confid.z, 1.0);
		}
		else if (idx >= vertex_confid_array.Size() && idx < vertex_confid_array.Size() + node_num) {
			const int offset = idx - vertex_confid_array.Size();
			const float4 node = reference_node_coordinates[offset];
			vertex = make_float4(node.x, node.y, node.z, 1.0);
		}

		//Keep priority queue using heap
		float d0 = 1e6f, d1 = 1e6f, d2 = 1e6f, d3 = 1e6f;
		unsigned short i0 = 0, i1 = 0, i2 = 0, i3 = 0;

		//Burte force
		bruteForceSearch4Padded(vertex, reference_node_coordinates, node_num, d0, d1, d2, d3, i0, i1, i2, i3);

		 //The forth part of vertex might be confidence
		const float3 v = make_float3(vertex.x, vertex.y, vertex.z);

		//Compute the knn weight given knn
		const float4 node0_v4 = reference_node_coordinates[i0];
		const float3 node0_v = make_float3(node0_v4.x, node0_v4.y, node0_v4.z);
		const float vn_dist0 = squared_norm(v - node0_v);

		const float4 node1_v4 = reference_node_coordinates[i1];
		const float3 node1_v = make_float3(node1_v4.x, node1_v4.y, node1_v4.z);
		const float vn_dist1 = squared_norm(v - node1_v);

		const float4 node2_v4 = reference_node_coordinates[i2];
		const float3 node2_v = make_float3(node2_v4.x, node2_v4.y, node2_v4.z);
		const float vn_dist2 = squared_norm(v - node2_v);

		const float4 node3_v4 = reference_node_coordinates[i3];
		const float3 node3_v = make_float3(node3_v4.x, node3_v4.y, node3_v4.z);
		const float vn_dist3 = squared_norm(v - node3_v);

		// Compute the weight of this node
		float4 weight;
		weight.x = __expf(-vn_dist0 / (2 * d_node_radius_square));
		weight.y = __expf(-vn_dist1 / (2 * d_node_radius_square));
		weight.z = __expf(-vn_dist2 / (2 * d_node_radius_square));
		weight.w = __expf(-vn_dist3 / (2 * d_node_radius_square));
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION) //Do a normalization on the weights?
		const float inv_weight_sum = 1.0f / fabsf_sum(weight);
		weight.x *= inv_weight_sum;
		weight.y *= inv_weight_sum;
		weight.z *= inv_weight_sum;
		weight.w *= inv_weight_sum;
#endif

		// Store the result to global memory
		if (idx < vertex_confid_array.Size()) 
		{
			vertex_knn_array[idx] = make_ushort4(i0, i1, i2, i3);
			vertex_knn_weight[idx] = weight;
		}
		else if (idx >= vertex_confid_array.Size() 
			&& idx < vertex_confid_array.Size() + node_num) 
		{
			const int offset = idx - vertex_confid_array.Size();
			node_knn_array[offset] = make_ushort4(i0, i1, i2, i3);
			node_knn_weight[offset] = weight;
		}
	}


	/* This kernel do skinning of vertex given the node
	 * and vertex coordinate, vertex.w can not be used
	 */
	__global__ void skinningVertexBruteForceKernel(
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
		float d0 = 1e6f, d1 = 1e6f, d2 = 1e6f, d3 = 1e6f;
		unsigned short i0 = 0, i1 = 0, i2 = 0, i3 = 0;

		//Burte force
		bruteForceSearch4Padded(vertex, reference_node_coordinates, node_num, d0, d1, d2, d3, i0, i1, i2, i3);

		 //The forth part of vertex might be confidence
		const float3 v = make_float3(vertex.x, vertex.y, vertex.z);

		//Compute the knn weight given knn
		const float4& node0_v4 = reference_node_coordinates[i0];
		const float3 node0_v = make_float3(node0_v4.x, node0_v4.y, node0_v4.z);
		const float vn_dist0 = squared_norm(v - node0_v);

		const float4& node1_v4 = reference_node_coordinates[i1];
		const float3 node1_v = make_float3(node1_v4.x, node1_v4.y, node1_v4.z);
		const float vn_dist1 = squared_norm(v - node1_v);

		const float4& node2_v4 = reference_node_coordinates[i2];
		const float3 node2_v = make_float3(node2_v4.x, node2_v4.y, node2_v4.z);
		const float vn_dist2 = squared_norm(v - node2_v);

		const float4& node3_v4 = reference_node_coordinates[i3];
		const float3 node3_v = make_float3(node3_v4.x, node3_v4.y, node3_v4.z);
		const float vn_dist3 = squared_norm(v - node3_v);

		// Compute the weight of this node
		float4 weight;
		weight.x = __expf(-vn_dist0 / (2 * d_node_radius_square));
		weight.y = __expf(-vn_dist1 / (2 * d_node_radius_square));
		weight.z = __expf(-vn_dist2 / (2 * d_node_radius_square));
		weight.w = __expf(-vn_dist3 / (2 * d_node_radius_square));
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION) //Do a normalization on the weights?
		const float inv_weight_sum = 1.0f / fabsf_sum(weight);
		weight.x *= inv_weight_sum;
		weight.y *= inv_weight_sum;
		weight.z *= inv_weight_sum;
		weight.w *= inv_weight_sum;
#endif

		//Write the nearest neighbour to storage
		nearest_neighbours[vertex_idx] = make_ushort4(i0, i1, i2, i3);
		knn_weight[vertex_idx] = weight;
	}//End of kernel

} // namespace device
} // namespace surfelwarp


/* The setup and allocation functions, all default
 */
surfelwarp::KNNBruteForceReferenceNodes::KNNBruteForceReferenceNodes() : m_num_nodes(0)
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
	
	//Clear it at first
	clearConstantPoints();
}

surfelwarp::KNNBruteForceReferenceNodes::Ptr surfelwarp::KNNBruteForceReferenceNodes::Instance() {
	static KNNBruteForceReferenceNodes::Ptr instance = nullptr;
	if(instance == nullptr) {
		instance.reset(new KNNBruteForceReferenceNodes());
	}
	return instance;
}

surfelwarp::KNNBruteForceReferenceNodes::~KNNBruteForceReferenceNodes() {
	m_invalid_nodes.release();
}

void surfelwarp::KNNBruteForceReferenceNodes::AllocateBuffer(unsigned max_num_points) {}

void surfelwarp::KNNBruteForceReferenceNodes::ReleaseBuffer() {}


/* Clear the constant node coordinates
 */
void surfelwarp::KNNBruteForceReferenceNodes::clearConstantPoints(cudaStream_t stream)
{
	cudaSafeCall(cudaMemcpyToSymbolAsync(
		device::reference_node_coordinates, 
		m_invalid_nodes.ptr(),
		sizeof(float4) * Constants::kMaxNumNodes, 
		0, // no offset
		cudaMemcpyDeviceToDevice,
		stream
	));
}

/* Build index copy the nodes into const memory
 */
void surfelwarp::KNNBruteForceReferenceNodes::BuildIndex(const DeviceArrayView<float4>& nodes, cudaStream_t stream)
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
		device::reference_node_coordinates, 
		nodes.RawPtr(),
		nodes.Size() * sizeof(float4), 
		0, // no offset
		cudaMemcpyDeviceToDevice, 
		stream
	));
	
	//Update size
	m_num_nodes = nodes.Size();
}

void surfelwarp::KNNBruteForceReferenceNodes::UpdateIndex(
	const SynchronizeArray<float4>& nodes, 
	size_t new_nodes_offset, 
	cudaStream_t stream
) {
	//Copy the value to device
	const auto node_view = std::move(nodes.DeviceArrayReadOnly());
	const auto new_node_size = node_view.Size() - new_nodes_offset;
	const float4* node_ptr = node_view.RawPtr() + new_nodes_offset;
	cudaSafeCall(cudaMemcpyToSymbolAsync(
		device::reference_node_coordinates, 
		node_ptr,
		new_node_size * sizeof(float4), 
		new_nodes_offset * sizeof(float4), 
		cudaMemcpyDeviceToDevice, 
		stream
	));
	
	//Update the size
	m_num_nodes = nodes.DeviceArraySize();
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	SURFELWARP_CHECK_EQ(nodes.DeviceArraySize(), nodes.HostArraySize());
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::KNNBruteForceReferenceNodes::replaceWithMorePoints(const DeviceArrayView<float4>& nodes, cudaStream_t stream)
{
	SURFELWARP_CHECK_GE(nodes.Size(), m_num_nodes) << "Please use BuildIndex() instead!";
	cudaSafeCall(cudaMemcpyToSymbolAsync(
		device::reference_node_coordinates, 
		nodes.RawPtr(),
		nodes.Size() * sizeof(float4), 
		0, // no offset
		cudaMemcpyDeviceToDevice, 
		stream
	));
	m_num_nodes = nodes.Size();
}

void surfelwarp::KNNBruteForceReferenceNodes::Skinning(
	const DeviceArrayView<float4>& vertex, 
	DeviceArraySlice<ushort4> knn, 
	DeviceArraySlice<float4> knn_weight, 
	cudaStream_t stream
) {
	dim3 blk(256);
	dim3 grid(divUp(vertex.Size(), blk.x));
	device::skinningVertexBruteForceKernel<<<grid, blk, 0, stream>>>(
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

void surfelwarp::KNNBruteForceReferenceNodes::Skinning(
	const DeviceArrayView<float4>& vertex, const DeviceArrayView<float4>& node, 
	DeviceArraySlice<ushort4> vertex_knn, DeviceArraySlice<ushort4> node_knn, 
	DeviceArraySlice<float4> vertex_knn_weight, DeviceArraySlice<float4> node_knn_weight,
	cudaStream_t stream
) {
	//Check the size
	SURFELWARP_CHECK_EQ(node.Size(), m_num_nodes);
	
	dim3 blk(256);
	dim3 grid(divUp(vertex.Size() + m_num_nodes, blk.x));
	device::skinningVertexNodeBruteForceKernel<<<grid, blk, 0, stream>>>(
		vertex,
		m_num_nodes,
		vertex_knn, vertex_knn_weight,
		node_knn, node_knn_weight
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}



