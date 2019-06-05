#include "common/Constants.h"
#include "core/geometry/WarpFieldExtender.h"

#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	/* Kernel and method for choosing node candidate from init knn array (not field)
	*/
	__global__ void labelVertexCandidateKernel(
		const DeviceArrayView<float4> vertex_confid_array,
		const ushort4* vertex_knn_array,
		const float4* node_coords_array,
		unsigned* vertex_candidate_label
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= vertex_confid_array.Size()) return;

		//Obtain vertex and its knn
		const float4 vertex_confid = vertex_confid_array[idx];
		const float4 vertex = make_float4(vertex_confid.x, vertex_confid.y, vertex_confid.z, 1.0);
		const ushort4 knn = vertex_knn_array[idx];

		//Check its distance to node 
		float4 node; float dist_square;
		bool covered = false;

		//knn-0
		node = node_coords_array[knn.x];
		dist_square = squared_norm_xyz(node - vertex);
		if (dist_square < d_node_radius_square) {
			covered = true;
		}

		//knn-1
		node = node_coords_array[knn.y];
		dist_square = squared_norm_xyz(node - vertex);
		if (dist_square < d_node_radius_square) {
			covered = true;
		}

		//knn-2
		node = node_coords_array[knn.z];
		dist_square = squared_norm_xyz(node - vertex);
		if (dist_square < d_node_radius_square) {
			covered = true;
		}

		//knn-3
		node = node_coords_array[knn.w];
		dist_square = squared_norm_xyz(node - vertex);
		if (dist_square < d_node_radius_square) {
			covered = true;
		}

		//Write it to output
		unsigned label = 1;
		if (covered) {
			label = 0;
		}
		vertex_candidate_label[idx] = label;
	}

	__global__ void compactCandidateKernel(
		const DeviceArrayView<unsigned> candidate_validity_label,
		const unsigned* prefixsum_validity_label,
		const float4* vertex_array,
		float4* valid_candidate_vertex
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= candidate_validity_label.Size()) return;
		if (candidate_validity_label[idx] > 0) {
			const float4 vertex = vertex_array[idx];
			valid_candidate_vertex[prefixsum_validity_label[idx] - 1] = make_float4(vertex.x, vertex.y, vertex.z, 1.0);
		}
	}

} // device
} // surfelwarp


void surfelwarp::WarpFieldExtender::labelCollectUncoveredNodeCandidate(
	const DeviceArrayView<float4>& vertex_array,
	const DeviceArrayView<ushort4>& vertex_knn,
	const DeviceArrayView<float4>& node_coordinates,
	cudaStream_t stream
) {
	m_candidate_validity_indicator.ResizeArrayOrException(vertex_array.Size());
	
	dim3 blk(64);
	dim3 grid(divUp(vertex_array.Size(), blk.x));
	device::labelVertexCandidateKernel<<<grid, blk, 0, stream>>>(
		vertex_array, vertex_knn.RawPtr(),
		node_coordinates.RawPtr(),
		m_candidate_validity_indicator.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Do a prefix sum
	SURFELWARP_CHECK(vertex_array.Size() == m_candidate_validity_indicator.ArraySize());
	m_validity_indicator_prefixsum.InclusiveSum(m_candidate_validity_indicator.ArrayView(), stream);
	
	//Do compaction
	device::compactCandidateKernel<<<grid, blk, 0, stream>>>(
		m_candidate_validity_indicator.ArrayView(),
		m_validity_indicator_prefixsum.valid_prefixsum_array.ptr(),
		vertex_array.RawPtr(),
		m_candidate_vertex_array.DevicePtr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::WarpFieldExtender::syncQueryUncoveredNodeCandidateSize(
	cudaStream_t stream
) {
	//Check the size
	const auto& prefixsum_array = m_validity_indicator_prefixsum.valid_prefixsum_array;
	SURFELWARP_CHECK(prefixsum_array.size() == m_candidate_validity_indicator.ArraySize());
	
	//The device ptr
	const unsigned* candidate_size_dev = prefixsum_array.ptr() + prefixsum_array.size() - 1;
	unsigned candidate_size;
	cudaSafeCall(cudaMemcpyAsync(
		&candidate_size,
		candidate_size_dev,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream
	));
	
	//Sync and check the size
	cudaSafeCall(cudaStreamSynchronize(stream));
	m_candidate_vertex_array.ResizeArrayOrException(candidate_size);
	if(candidate_size != 0)
		m_candidate_vertex_array.SynchronizeToHost(stream, true);
	
	//Debug method
	//LOG(INFO) << "The number of node candidates is " << m_candidate_vertex_array.DeviceArraySize();
}


