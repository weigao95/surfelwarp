//
// Created by wei on 5/7/18.
//

#include "common/macro_utils.h"
#include "core/geometry/SurfelNodeDeformer.h"
#include <device_launch_parameters.h>


namespace surfelwarp { namespace device {
	
	__global__ void forwardWarpVertexAndNodeKernel(
		const DeviceArrayView<float4> canonical_vertex_confid,
		const float4* canonical_normal_radius,
		const ushort4* vertex_knn_array, const float4* vertex_knn_weight,
		const DeviceArrayView<float4> canonical_nodes_coordinate,
		const ushort4* node_knn_array, const float4* node_knn_weight,
		const DualQuaternion* warp_field,
		//Output array, shall be size correct
		float4* live_vertex_confid,
		float4* live_normal_radius,
		float4* live_node_coordinate
	) {
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;
		ushort4 knn; float4 weight;
		float4  vertex;
		float4 normal = make_float4(0, 0, 0, 0);
		if (idx < canonical_vertex_confid.Size()) 
		{
			knn = vertex_knn_array[idx];
			weight = vertex_knn_weight[idx];
			vertex = canonical_vertex_confid[idx];
			normal = canonical_normal_radius[idx];
		}
		else if (
			idx >= canonical_vertex_confid.Size() && 
			idx < canonical_vertex_confid.Size() + canonical_nodes_coordinate.Size()) 
		{
			const int offset = idx - canonical_vertex_confid.Size();
			knn = node_knn_array[offset];
			weight = node_knn_weight[offset];
			vertex = canonical_nodes_coordinate[offset];
		}

		//Do warpping
		DualQuaternion dq_average = averageDualQuaternion(warp_field, knn, weight);
		const mat34 se3 = dq_average.se3_matrix();
		float3 v3 = make_float3(vertex.x, vertex.y, vertex.z);
		float3 n3 = make_float3(normal.x, normal.y, normal.z);
		v3 = se3.rot * v3 + se3.trans;
		n3 = se3.rot * n3;
		vertex = make_float4(v3.x, v3.y, v3.z, vertex.w);
		normal = make_float4(n3.x, n3.y, n3.z, normal.w);

		//Save it
		if (idx < canonical_vertex_confid.Size()) 
		{
			live_vertex_confid[idx] = vertex;
			live_normal_radius[idx] = normal;
		}
		else if (
			idx >= canonical_vertex_confid.Size() && 
			idx < canonical_vertex_confid.Size() + canonical_nodes_coordinate.Size()) 
		{
			const int offset = idx - canonical_vertex_confid.Size();
			live_node_coordinate[offset] = vertex;
		}
	}


	__global__ void inverseWarpVertexNormalKernel(
		const DeviceArrayView<float4> live_vertex_confid_array,
		const float4* live_normal_radius_array,
		const ushort4* vertex_knn_array,
		const float4* vertex_knn_weight,
		const DualQuaternion* device_warp_field,
		float4* canonical_vertex_confid,
		float4* canonical_normal_radius
	) {
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < live_vertex_confid_array.Size()) {
			const float4 live_vertex_confid = live_vertex_confid_array[idx];
			const float4 live_normal_radius = live_normal_radius_array[idx];
			const ushort4 knn = vertex_knn_array[idx];
			const float4 knn_weight = vertex_knn_weight[idx];
			auto dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
			mat34 se3 = dq_average.se3_matrix();
			float3 vertex = make_float3(live_vertex_confid.x, live_vertex_confid.y, live_vertex_confid.z);
			float3 normal = make_float3(live_normal_radius.x, live_normal_radius.y, live_normal_radius.z);
			//Apply the inversed warping without construction of the matrix
			vertex = se3.apply_inversed_se3(vertex);
			normal = se3.rot.transpose_dot(normal);
			canonical_vertex_confid[idx] = make_float4(vertex.x, vertex.y, vertex.z, live_vertex_confid.w);
			canonical_normal_radius[idx] = make_float4(normal.x, normal.y, normal.z, live_normal_radius.w);
		}
	}


} // device
} // surfelwarp

void surfelwarp::SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(
	WarpField & warp_field, 
	SurfelGeometry & geometry, 
	const DeviceArrayView<DualQuaternion>& node_se3, 
	cudaStream_t stream
) {
	//Check the size
	CheckSurfelGeometySize(geometry);

	//The node se3 should have the same size
	SURFELWARP_CHECK(node_se3.Size() == warp_field.m_node_se3.DeviceArraySize());

	//Update the size of live nodes
	warp_field.m_live_node_coords.ResizeArrayOrException(warp_field.m_reference_node_coords.DeviceArraySize());
	SURFELWARP_CHECK_EQ(warp_field.m_node_knn.ArraySize(), warp_field.m_live_node_coords.ArraySize());
	SURFELWARP_CHECK_EQ(warp_field.m_node_knn_weight.ArraySize(), warp_field.m_live_node_coords.ArraySize());

	//Load the data
	const auto reference_vertex = geometry.m_reference_vertex_confid.ArrayView();
	const auto reference_normal = geometry.m_reference_normal_radius.ArrayView();
	const auto knn = geometry.m_surfel_knn.ArrayView();
	const auto knn_weight = geometry.m_surfel_knn_weight.ArrayView();
	auto live_vertex = geometry.m_live_vertex_confid.ArraySlice();
	auto live_normal = geometry.m_live_normal_radius.ArraySlice();

	//Invoke kernel
	dim3 blk(256);
	dim3 grid(divUp(reference_vertex.Size() + warp_field.m_reference_node_coords.DeviceArraySize(), blk.x));
	device::forwardWarpVertexAndNodeKernel<<<grid, blk, 0, stream>>>(
		reference_vertex, 
		reference_normal, 
		knn, 
		knn_weight, 
		//For nodes
		warp_field.m_reference_node_coords.DeviceArrayReadOnly(), 
		warp_field.m_node_knn.Ptr(), warp_field.m_node_knn_weight.Ptr(),
		node_se3.RawPtr(), 
		//Output
		live_vertex, 
		live_normal, 
		warp_field.m_live_node_coords.Ptr()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(
	surfelwarp::WarpField &warp_field,
	surfelwarp::SurfelGeometry &geometry,
	cudaStream_t stream
) {
	ForwardWarpSurfelsAndNodes(
		warp_field, 
		geometry,
		warp_field.m_node_se3.DeviceArrayReadOnly(), 
		stream
	);
}


void surfelwarp::SurfelNodeDeformer::InverseWarpSurfels(
	SurfelGeometry &geometry,
	const DeviceArrayView<DualQuaternion> &node_se3,
	cudaStream_t stream
) {
	//Check the size
	CheckSurfelGeometySize(geometry);
	
	//Load the data
	const auto live_vertex = geometry.m_live_vertex_confid.ArrayView();
	const auto live_normal = geometry.m_live_normal_radius.ArrayView();
	const auto knn = geometry.m_surfel_knn.ArrayView();
	const auto knn_weight = geometry.m_surfel_knn_weight.ArrayView();
	auto reference_vertex = geometry.m_reference_vertex_confid.ArraySlice();
	auto reference_normal = geometry.m_reference_normal_radius.ArraySlice();
	
	//Do warping
	dim3 blk(256);
	dim3 grid(divUp(reference_vertex.Size(), blk.x));
	device::inverseWarpVertexNormalKernel<<<grid, blk, 0, stream>>>(
		live_vertex,
			live_normal,
			knn, knn_weight,
			node_se3.RawPtr(),
			reference_vertex,
			reference_normal
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::SurfelNodeDeformer::InverseWarpSurfels(
	const WarpField & warp_field, 
	SurfelGeometry & geometry, 
	const DeviceArrayView<DualQuaternion>& node_se3, 
	cudaStream_t stream
) {
	//The node se3 should have the same size
	SURFELWARP_CHECK(node_se3.Size() == warp_field.m_node_se3.DeviceArraySize());
	InverseWarpSurfels(geometry, node_se3, stream);
}


void surfelwarp::SurfelNodeDeformer::InverseWarpSurfels(
	const surfelwarp::WarpField &warp_field,
	surfelwarp::SurfelGeometry &geometry,
	cudaStream_t stream
) {
	//Check the size
	InverseWarpSurfels(
		warp_field, 
		geometry, 
		warp_field.m_node_se3.DeviceArrayReadOnly(), 
		stream
	);
}

void surfelwarp::SurfelNodeDeformer::CheckSurfelGeometySize(const SurfelGeometry &geometry) {
	const auto num_surfels = geometry.NumValidSurfels();
	SURFELWARP_CHECK(geometry.m_reference_vertex_confid.ArraySize() == num_surfels);
	SURFELWARP_CHECK(geometry.m_reference_normal_radius.ArraySize() == num_surfels);
	SURFELWARP_CHECK(geometry.m_surfel_knn.ArraySize() == num_surfels);
	SURFELWARP_CHECK(geometry.m_surfel_knn_weight.ArraySize() == num_surfels);
	SURFELWARP_CHECK(geometry.m_live_vertex_confid.ArraySize() == num_surfels);
	SURFELWARP_CHECK(geometry.m_live_normal_radius.ArraySize() == num_surfels);
}
