#include "common/Constants.h"
#include "core/geometry/NodeGraphBuilderNaive.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__device__ __forceinline__ float distance_square(const float4& p1, const float4& p2) {
		return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z);
	}

	enum {
		//The default nn size of nodes
		d_node_nn_size = 8
	};

	/* The node graph is the neighbour of a control node, each node has 8 neighbour,
	which is stored as consequential elements in the node graph. The node are in
	the number of thousand level, thus iterate only over nodes are not very expensive
	*/
	__global__ void buildNaiveNodeGraphKernel(
		const DeviceArrayView<float4> node_coords,
		ushort2* device_node_graph
	) {
		const int node_num = node_coords.Size();
		const int idx = blockDim.x * blockIdx.x + threadIdx.x;
		if (idx >= node_num) return;
		float dist_vec[d_node_nn_size];
		int idx_vec[d_node_nn_size];
		//First init these values
		for (int k = 0; k < d_node_nn_size; k++) {
			idx_vec[k] = -1;
			dist_vec[k] = 1e5;
		}

		//Perform brute-force search of these nodes
		const float4 p_idx = node_coords[idx];
		int max_index = 0;
		for (int k = 0; k < node_num; k++) {
			const float4 coord = node_coords[k];
			const float new_dist = distance_square(p_idx, coord);
			if (new_dist > 1e-6 && new_dist < dist_vec[max_index]) {
				dist_vec[max_index] = new_dist;
				idx_vec[max_index] = k;

				//Rechoice the index with the maximum distance
				max_index = 0;
				float max_dist = 0;
				for (int j = 0; j < d_node_nn_size; j++) {
					if (dist_vec[j] > max_dist) {
						max_index = j;
						max_dist = dist_vec[j];
					}
				}
			}
		}

		//Record the computed distance on the ptr
		for (int k = 0; k < d_node_nn_size; k++) {
			const int offset = idx * d_node_nn_size + k;
			device_node_graph[offset] = make_ushort2(idx, idx_vec[k]);
		}
	}


} // namespace device
} // namespace surfelwarp

void surfelwarp::NodeGraphBuilderNaive::buildNodeGraph(
	const DeviceArrayView<float4>& reference_nodes, 
	DeviceArraySlice<ushort2> node_graph, 
	cudaStream_t stream
) {
	dim3 blk(64);
	dim3 grid(divUp(reference_nodes.Size(), blk.x));
	device::buildNaiveNodeGraphKernel<<<grid, blk, 0, stream>>>(
		reference_nodes, 
		node_graph.RawPtr()
	);
}


void surfelwarp::NodeGraphBuilderNaive::BuildNodeGraph(
	const DeviceArrayView<float4>& reference_nodes, 
	DeviceBufferArray<ushort2>& node_graph, 
	cudaStream_t stream
) {
	node_graph.ResizeArrayOrException(reference_nodes.Size() * Constants::kNumNodeGraphNeigbours);
	buildNodeGraph(reference_nodes, node_graph.ArraySlice(), stream);
}