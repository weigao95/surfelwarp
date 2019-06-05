#include "common/Constants.h"
#include "math/vector_ops.hpp"
#include "core/geometry/ReferenceNodeSkinner.h"
#include "core/geometry/brute_foce_knn.cuh"


#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	__device__ __constant__ float4 reference_node_coordinates[d_max_num_nodes];

	/* This kernel do skinning of both vertex and nodes given
	 * node coordinate and vertex coordinate, vertex.w can not be used
	 */
	__global__ void skinningVertexAndNodeBruteForceKernel(
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
		float4 distance = make_float4(1e6f, 1e6f, 1e6f, 1e6f);
		ushort4 node_idx = make_ushort4(0, 0, 0, 0);

		//Burte force
		bruteForceSearch4Padded(vertex, reference_node_coordinates, node_num, distance, node_idx);

		 //The forth part of vertex might be confidence
		const float3 v = make_float3(vertex.x, vertex.y, vertex.z);

		//Compute the knn weight given knn
		const float4 node0_v4 = reference_node_coordinates[node_idx.x];
		const float3 node0_v = make_float3(node0_v4.x, node0_v4.y, node0_v4.z);
		const float vn_dist0 = squared_norm(v - node0_v);

		const float4 node1_v4 = reference_node_coordinates[node_idx.y];
		const float3 node1_v = make_float3(node1_v4.x, node1_v4.y, node1_v4.z);
		const float vn_dist1 = squared_norm(v - node1_v);

		const float4 node2_v4 = reference_node_coordinates[node_idx.z];
		const float3 node2_v = make_float3(node2_v4.x, node2_v4.y, node2_v4.z);
		const float vn_dist2 = squared_norm(v - node2_v);

		const float4 node3_v4 = reference_node_coordinates[node_idx.w];
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
			vertex_knn_array[idx] = node_idx;
			vertex_knn_weight[idx] = weight;
		}
		else if (idx >= vertex_confid_array.Size() 
			&& idx < vertex_confid_array.Size() + node_num) 
		{
			const int offset = idx - vertex_confid_array.Size();
			node_knn_array[offset] = node_idx;
			node_knn_weight[offset] = weight;
		}
	}



	__global__ void updateVertexNodeKnnWeightKernel(
		const DeviceArrayView<float4> vertex_confid_array,
		ushort4* vertex_knn_array, float4* vertex_knn_weight,
		DeviceArraySlice<ushort4> node_knn_array, float4* node_knn_weight,
		// The offset and number of added nodes
		const int node_offset, const int padded_node_num
	) {
		// Outof bound: for both vertex and node knn are updated by this kernel
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= vertex_confid_array.Size() + node_knn_array.Size()) return;

		// Collect information form global memory
		float3 v;
		ushort4 knn;
		if (idx < vertex_confid_array.Size()) {
			const float4 vertex_confid = vertex_confid_array[idx];
			v = make_float3(vertex_confid.x, vertex_confid.y, vertex_confid.z);
			knn = vertex_knn_array[idx];
		}
		else if (idx >= vertex_confid_array.Size() && idx < vertex_confid_array.Size() + node_knn_array.Size()) {
			const auto offset = idx - vertex_confid_array.Size();
			const float4 node = reference_node_coordinates[offset];
			v = make_float3(node.x, node.y, node.z);
			knn = node_knn_array[offset];
		}
		else {
			return;
		}

		// load knn for each thread
		const ushort4 knn_prev = knn;
		float4 n0 = reference_node_coordinates[knn.x];
		float tmp0 = v.x - n0.x;
		float tmp1 = v.y - n0.y;
		float tmp2 = v.z - n0.z;

		float4 n1 = reference_node_coordinates[knn.y];
		float tmp6 = v.x - n1.x;
		float tmp7 = v.y - n1.y;
		float tmp8 = v.z - n1.z;

		float4 n2 = reference_node_coordinates[knn.z];
		float tmp12 = v.x - n2.x;
		float tmp13 = v.y - n2.y;
		float tmp14 = v.z - n2.z;

		float4 n3 = reference_node_coordinates[knn.w];
		float tmp18 = v.x - n3.x;
		float tmp19 = v.y - n3.y;
		float tmp20 = v.z - n3.z;

		float tmp3 = __fmul_rn(tmp0, tmp0);
		float tmp9 = __fmul_rn(tmp6, tmp6);
		float tmp15 = __fmul_rn(tmp12, tmp12);
		float tmp21 = __fmul_rn(tmp18, tmp18);

		float tmp4 = __fmaf_rn(tmp1, tmp1, tmp3);
		float tmp10 = __fmaf_rn(tmp7, tmp7, tmp9);
		float tmp16 = __fmaf_rn(tmp13, tmp13, tmp15);
		float tmp22 = __fmaf_rn(tmp19, tmp19, tmp21);

		float tmp5 = __fmaf_rn(tmp2, tmp2, tmp4);
		float tmp11 = __fmaf_rn(tmp8, tmp8, tmp10);
		float tmp17 = __fmaf_rn(tmp14, tmp14, tmp16);
		float tmp23 = __fmaf_rn(tmp20, tmp20, tmp22);

		//keep priority queue using heap
		float4 distance = make_float4(tmp5, tmp11, tmp17, tmp23);
		KnnHeapDevice heap(distance, knn);

		//The update loop
		for (auto k = node_offset; k < padded_node_num + node_offset; k += 4) {
			n0 = reference_node_coordinates[k + 0];
			tmp0 = v.x - n0.x;
			tmp1 = v.y - n0.y;
			tmp2 = v.z - n0.z;

			n1 = reference_node_coordinates[k + 1];
			tmp6 = v.x - n1.x;
			tmp7 = v.y - n1.y;
			tmp8 = v.z - n1.z;

			n2 = reference_node_coordinates[k + 2];
			tmp12 = v.x - n2.x;
			tmp13 = v.y - n2.y;
			tmp14 = v.z - n2.z;

			n3 = reference_node_coordinates[k + 3];
			tmp18 = v.x - n3.x;
			tmp19 = v.y - n3.y;
			tmp20 = v.z - n3.z;

			tmp3 = __fmul_rn(tmp0, tmp0);
			tmp9 = __fmul_rn(tmp6, tmp6);
			tmp15 = __fmul_rn(tmp12, tmp12);
			tmp21 = __fmul_rn(tmp18, tmp18);

			tmp4 = __fmaf_rn(tmp1, tmp1, tmp3);
			tmp10 = __fmaf_rn(tmp7, tmp7, tmp9);
			tmp16 = __fmaf_rn(tmp13, tmp13, tmp15);
			tmp22 = __fmaf_rn(tmp19, tmp19, tmp21);

			tmp5 = __fmaf_rn(tmp2, tmp2, tmp4);
			tmp11 = __fmaf_rn(tmp8, tmp8, tmp10);
			tmp17 = __fmaf_rn(tmp14, tmp14, tmp16);
			tmp23 = __fmaf_rn(tmp20, tmp20, tmp22);

			//Update it
			heap.update(k + 0, tmp5);
			heap.update(k + 1, tmp11);
			heap.update(k + 2, tmp17);
			heap.update(k + 3, tmp23);
		}//End of the update loop
		
		 // If the knn doesn't change
		if (knn.x == knn_prev.x && knn.y == knn_prev.y && knn.z == knn_prev.z && knn.w == knn_prev.w) return;

		// If changed, update the weight
		const float4 node0_v4 = reference_node_coordinates[knn.x];
		const float3 node0_v = make_float3(node0_v4.x, node0_v4.y, node0_v4.z);
		const float vn_dist0 = squared_norm(v - node0_v);

		const float4 node1_v4 = reference_node_coordinates[knn.y];
		const float3 node1_v = make_float3(node1_v4.x, node1_v4.y, node1_v4.z);
		const float vn_dist1 = squared_norm(v - node1_v);

		const float4 node2_v4 = reference_node_coordinates[knn.z];
		const float3 node2_v = make_float3(node2_v4.x, node2_v4.y, node2_v4.z);
		const float vn_dist2 = squared_norm(v - node2_v);

		const float4 node3_v4 = reference_node_coordinates[knn.w];
		const float3 node3_v = make_float3(node3_v4.x, node3_v4.y, node3_v4.z);
		const float vn_dist3 = squared_norm(v - node3_v);

		// Compute the weight of this node
		float4 weight;
		weight.x = __expf(-vn_dist0 / (2 * d_node_radius_square));
		weight.y = __expf(-vn_dist1 / (2 * d_node_radius_square));
		weight.z = __expf(-vn_dist2 / (2 * d_node_radius_square));
		weight.w = __expf(-vn_dist3 / (2 * d_node_radius_square));

		//Do a normalization?
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION)
		const float weight_sum = weight.x + weight.y + weight.z + weight.w;
		const float inv_weight_sum = 1.0f / weight_sum;
		weight.x *= inv_weight_sum;
		weight.y *= inv_weight_sum;
		weight.z *= inv_weight_sum;
		weight.w *= inv_weight_sum;
#endif

		// Store the result to global memory
		if (idx < vertex_confid_array.Size()) {
			vertex_knn_array[idx] = knn;
			vertex_knn_weight[idx] = weight;
		}
		else if (idx >= vertex_confid_array.Size() && idx < vertex_confid_array.Size() + node_knn_array.Size()) {
			const int offset = idx - vertex_confid_array.Size();
			node_knn_array[offset] = knn;
			node_knn_weight[offset] = weight;
		}

	} // End of kernel

} // device
} // surfelwarp

surfelwarp::ReferenceNodeSkinner::ReferenceNodeSkinner() {
	m_init_skinner = nullptr; //Just use brute force at first
	m_num_bruteforce_nodes = 0;
	
	//Update the invalid nodes
	m_invalid_nodes.create(Constants::kMaxNumNodes);
	
	//The other part of the constant memory should be filled with invalid points
	std::vector<float4> h_invalid_nodes;
	h_invalid_nodes.resize(Constants::kMaxNumNodes);
	float* begin = (float*)h_invalid_nodes.data();
	float* end = begin + 4 * Constants::kMaxNumNodes;
	std::fill(begin, end, 1e6f);
	m_invalid_nodes.upload(h_invalid_nodes);
	
	//Fill the constant memory with invalid values at first
	fillInvalidConstantPoints();
}

/* The method for initial skinning
 */
void surfelwarp::ReferenceNodeSkinner::BuildInitialSkinningIndex(const SynchronizeArray<float4>& nodes, cudaStream_t stream)
{
	//Build the index for brute force searcher
	buildBruteForceIndex(nodes.DeviceArrayReadOnly(), stream);
	SURFELWARP_CHECK(m_num_bruteforce_nodes == nodes.DeviceArraySize());
	
	//If there is a customized searcher
	if(m_init_skinner != nullptr) {
		m_init_skinner->BuildIndexHostNodes(nodes.HostArray(), stream);
	}
}


void surfelwarp::ReferenceNodeSkinner::fillInvalidConstantPoints(cudaStream_t stream) {
	cudaSafeCall(cudaMemcpyToSymbolAsync(
		device::reference_node_coordinates,
		m_invalid_nodes.ptr(),
		sizeof(float4) * Constants::kMaxNumNodes,
		0, // no offset
		cudaMemcpyDeviceToDevice,
		stream
	));
}

void surfelwarp::ReferenceNodeSkinner::replaceWithMorePoints(
	const DeviceArrayView<float4> &nodes,
	cudaStream_t stream
) {
	SURFELWARP_CHECK_GE(nodes.Size(), m_num_bruteforce_nodes) << "Please use BuildIndex() instead!";
	cudaSafeCall(cudaMemcpyToSymbolAsync(
		device::reference_node_coordinates,
		nodes.RawPtr(),
		nodes.Size() * sizeof(float4),
		0, // no offset
		cudaMemcpyDeviceToDevice,
		stream
	));
	m_num_bruteforce_nodes = nodes.Size();
}

void surfelwarp::ReferenceNodeSkinner::buildBruteForceIndex(
	const DeviceArrayView<float4> &nodes,
	cudaStream_t stream
) {
	//If the new nodes is more than previous nodes
	if(nodes.Size() >= m_num_bruteforce_nodes) {
		replaceWithMorePoints(nodes, stream);
		return;
	}
	
	//Check the size
	SURFELWARP_CHECK(nodes.Size() <= Constants::kMaxNumNodes) << "Too many nodes";
	
	//First clear the buffer
	fillInvalidConstantPoints(stream);
	
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
	m_num_bruteforce_nodes = nodes.Size();
}

void surfelwarp::ReferenceNodeSkinner::performBruteForceSkinning(
	const DeviceArrayView<float4>& reference_vertex, 
	const DeviceArrayView<float4>& reference_node, 
	DeviceArraySlice<ushort4> vertex_knn, 
	DeviceArraySlice<ushort4> node_knn, 
	DeviceArraySlice<float4> vertex_knn_weight, 
	DeviceArraySlice<float4> node_knn_weight, 
	cudaStream_t stream
) const {
	//Check the size
	SURFELWARP_CHECK_EQ(reference_node.Size(), m_num_bruteforce_nodes);
	
	dim3 blk(256);
	dim3 grid(divUp(reference_vertex.Size() + m_num_bruteforce_nodes, blk.x));
	device::skinningVertexAndNodeBruteForceKernel<<<grid, blk, 0, stream>>>(
		reference_vertex,
		m_num_bruteforce_nodes,
		vertex_knn, vertex_knn_weight,
		node_knn, node_knn_weight
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::ReferenceNodeSkinner::PerformSkinning(
	SurfelGeometry::SkinnerInput geometry,
	WarpField::SkinnerInput warp_field,
	cudaStream_t stream
) {
	if(m_init_skinner != nullptr) { //Using the customized version
	
	} else {
		//Using brute for skinning here
		performBruteForceSkinning(
			geometry.reference_vertex_confid,
			warp_field.reference_node_coords,
			geometry.surfel_knn, warp_field.node_knn,
			geometry.surfel_knn_weight, warp_field.node_knn_weight,
			stream
		);
	}
	
	//Check it
	/*KNNSearch::CheckKNNSearch(
		warp_field.reference_node_coords,
		warp_field.reference_node_coords,
		warp_field.node_knn.ArrayView()
	);
	KNNSearch::CheckKNNSearch(
		warp_field.reference_node_coords,
		geometry.reference_vertex_confid,
		geometry.surfel_knn.ArrayView()
	);*/
}


/* The method for skinning update. nodes[newnode_offset] should be the first new node
 */
void surfelwarp::ReferenceNodeSkinner::UpdateBruteForceSkinningIndexWithNewNodes(
	const DeviceArrayView<float4>& nodes, 
	unsigned newnode_offset, 
	cudaStream_t stream
) {
	//Check the size
	const unsigned prev_nodesize = newnode_offset;
	SURFELWARP_CHECK(nodes.Size() >= prev_nodesize); //There should be more nodes now
	SURFELWARP_CHECK(nodes.Size() <= Constants::kMaxNumNodes);
	
	//There is no node to append
	if(nodes.Size() == prev_nodesize) return;
	
	//Everything seems to be correct, do it
	const auto new_node_size = nodes.Size() - newnode_offset;
	const float4* node_ptr = nodes.RawPtr() + newnode_offset;
	cudaSafeCall(cudaMemcpyToSymbolAsync(
		device::reference_node_coordinates,
		node_ptr,
		new_node_size * sizeof(float4),
		newnode_offset * sizeof(float4),
		cudaMemcpyDeviceToDevice,
		stream
	));
	
	//Update the size
	m_num_bruteforce_nodes = nodes.Size();
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::ReferenceNodeSkinner::updateSkinning(
	unsigned newnode_offset, 
	const DeviceArrayView<float4>& reference_vertex, 
	const DeviceArrayView<float4>& reference_node, 
	DeviceArraySlice<ushort4> vertex_knn, 
	DeviceArraySlice<ushort4> node_knn, 
	DeviceArraySlice<float4> vertex_knn_weight, 
	DeviceArraySlice<float4> node_knn_weight, 
	cudaStream_t stream
) const {
	//Check the size
	const unsigned prev_nodesize = newnode_offset;
	SURFELWARP_CHECK(reference_node.Size() >= prev_nodesize); //There should be more nodes now
	if(reference_node.Size() == prev_nodesize) return;
	
	//The index should be updated
	SURFELWARP_CHECK(m_num_bruteforce_nodes == reference_node.Size()) << "The index is not updated";

	//The numer of append node
	const auto num_appended_node = m_num_bruteforce_nodes - newnode_offset;
	const auto padded_newnode_num = divUp(num_appended_node, 4) * 4;

	//Let's to it
	dim3 blk(256);
	dim3 grid(divUp(reference_vertex.Size() + m_num_bruteforce_nodes, blk.x));
	device::updateVertexNodeKnnWeightKernel<<<grid, blk, 0, stream>>>(
		reference_vertex, 
		vertex_knn.RawPtr(), vertex_knn_weight.RawPtr(), 
		node_knn, node_knn_weight.RawPtr(), 
		newnode_offset, padded_newnode_num
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::ReferenceNodeSkinner::PerformSkinningUpdate(
	SurfelGeometry::SkinnerInput geometry,
	WarpField::SkinnerInput warp_field,
	unsigned newnode_offset,
	cudaStream_t stream
) {
	//Check the size
	SURFELWARP_CHECK(geometry.surfel_knn.Size() == geometry.surfel_knn_weight.Size());
	SURFELWARP_CHECK(geometry.surfel_knn.Size() == geometry.reference_vertex_confid.Size());
	SURFELWARP_CHECK(warp_field.reference_node_coords.Size() == warp_field.node_knn.Size());
	SURFELWARP_CHECK(warp_field.reference_node_coords.Size() == warp_field.node_knn_weight.Size());
	
	//Hand in to workforce
	updateSkinning(
		newnode_offset,
		geometry.reference_vertex_confid,
		warp_field.reference_node_coords,
		geometry.surfel_knn, warp_field.node_knn,
		geometry.surfel_knn_weight, warp_field.node_knn_weight,
		stream
	);
	
	//Check it
	/*KNNSearch::CheckApproximateKNNSearch(
		warp_field.reference_node_coords,
		geometry.reference_vertex_confid,
		geometry.surfel_knn.ArrayView()
	);
	KNNSearch::CheckKNNSearch(
		warp_field.reference_node_coords,
		warp_field.reference_node_coords,
		warp_field.node_knn.ArrayView()
	);*/
}



