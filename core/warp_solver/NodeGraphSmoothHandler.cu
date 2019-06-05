#include "core/warp_solver/NodeGraphSmoothHandler.h"
#include "common/Constants.h"
#include "core/warp_solver/solver_constants.h"

#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	__global__ void forwardWarpSmootherNodeKernel(
		DeviceArrayView<ushort2> node_graph,
		const float4* reference_node_array,
		const DualQuaternion* node_se3,
		float3* Ti_xj_array,
		float3* Tj_xj_array,
		unsigned char* validity_indicator_array
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(idx < node_graph.Size()) {
			const ushort2 node_ij = node_graph[idx];
			const auto xi = reference_node_array[node_ij.x];
			const auto xj = reference_node_array[node_ij.y];
			DualQuaternion dq_i = node_se3[node_ij.x];
			DualQuaternion dq_j = node_se3[node_ij.y];
			const mat34 Ti = dq_i.se3_matrix();
			const mat34 Tj = dq_j.se3_matrix();

			const auto Ti_xj = Ti.rot * xj + Ti.trans;
			const auto Tj_xj = Tj.rot * xj + Tj.trans;
			unsigned char validity_indicator = 1;
#if defined(CLIP_FARAWAY_NODEGRAPH_PAIR)
			if (squared_norm_xyz(xi - xj) > 64 * d_node_radius_square) {
				validity_indicator = 0;
			}
#endif
			//Save all the data
			Ti_xj_array[idx] = Ti_xj;
			Tj_xj_array[idx] = Tj_xj;
			validity_indicator_array[idx] = validity_indicator;
		}
	}


} // device
} // surfelwarp


surfelwarp::NodeGraphSmoothHandler::NodeGraphSmoothHandler() {
	const auto num_smooth_terms = Constants::kMaxNumNodes * Constants::kNumNodeGraphNeigbours;
	Ti_xj_.AllocateBuffer(num_smooth_terms);
	Tj_xj_.AllocateBuffer(num_smooth_terms);
	m_pair_validity_indicator.AllocateBuffer(num_smooth_terms);
}

surfelwarp::NodeGraphSmoothHandler::~NodeGraphSmoothHandler() {
	Ti_xj_.ReleaseBuffer();
	Tj_xj_.ReleaseBuffer();
	m_pair_validity_indicator.ReleaseBuffer();
}

void surfelwarp::NodeGraphSmoothHandler::SetInputs(
	const DeviceArrayView<DualQuaternion>& node_se3, 
	const DeviceArrayView<ushort2>& node_graph, 
	const DeviceArrayView<float4>& reference_nodes
) {
	m_node_se3 = node_se3;
	m_node_graph = node_graph;
	m_reference_node_coords = reference_nodes;
}



/* The method to build the term2jacobian
 */
void surfelwarp::NodeGraphSmoothHandler::forwardWarpSmootherNodes(cudaStream_t stream) {
	Ti_xj_.ResizeArrayOrException(m_node_graph.Size());
	Tj_xj_.ResizeArrayOrException(m_node_graph.Size());
	m_pair_validity_indicator.ResizeArrayOrException(m_node_graph.Size());

	dim3 blk(128);
	dim3 grid(divUp(m_node_graph.Size(), blk.x));
	device::forwardWarpSmootherNodeKernel<<<grid, blk, 0, stream>>>(
		m_node_graph, 
		m_reference_node_coords.RawPtr(),
		m_node_se3.RawPtr(), 
		Ti_xj_.Ptr(), Tj_xj_.Ptr(), 
		m_pair_validity_indicator.Ptr()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::NodeGraphSmoothHandler::BuildTerm2Jacobian(cudaStream_t stream) {
	forwardWarpSmootherNodes(stream);
}

surfelwarp::NodeGraphSmoothTerm2Jacobian surfelwarp::NodeGraphSmoothHandler::Term2JacobianMap() const
{
	NodeGraphSmoothTerm2Jacobian map;
	map.node_se3 = m_node_se3;
	map.reference_node_coords = m_reference_node_coords;
	map.node_graph = m_node_graph;
	map.Ti_xj = Ti_xj_.ArrayView();
	map.Tj_xj = Tj_xj_.ArrayView();
	map.validity_indicator = m_pair_validity_indicator.ArrayView();
	return map;
}