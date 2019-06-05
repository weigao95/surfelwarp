#include "common/logging.h"
#include "core/WarpField.h"
#include "math/DualQuaternion.hpp"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {


	__global__ void forwardWarpVertexDebugKernel(
		const DeviceArrayView<float4> canonical_vertex_confid,
		const ushort4* vertex_knn_array, const float4* vertex_knn_weight,
		const DualQuaternion* warp_field,
		//Output array, shall be size correct
		float4* live_vertex_confid
	) {
		const int idx = threadIdx.x + blockDim.x * blockIdx.x;
		ushort4 knn; float4 weight;
		float4  vertex;
		if (idx < canonical_vertex_confid.Size()) 
		{
			knn = vertex_knn_array[idx];
			weight = vertex_knn_weight[idx];
			vertex = canonical_vertex_confid[idx];
		}

		//Do warping
		DualQuaternion dq_average = averageDualQuaternion(warp_field, knn, weight);
		const mat34 se3 = dq_average.se3_matrix();
		float3 v3 = make_float3(vertex.x, vertex.y, vertex.z);
		v3 = se3.rot * v3 + se3.trans;
		vertex = make_float4(v3.x, v3.y, v3.z, vertex.w);

		//Save it
		if (idx < canonical_vertex_confid.Size()) {
			live_vertex_confid[idx] = vertex;
		}
	}


} // namespace device
} // namespace surfelwarp


void surfelwarp::WarpField::ForwardWarpDebug(
	const DeviceArrayView<float4>& reference_vertex, 
	const DeviceArrayView<ushort4>& knn, 
	const DeviceArrayView<float4>& knn_weight, 
	const DeviceArrayView<DualQuaternion>& node_se3, 
	DeviceArraySlice<float4> live_vertex
) const {
	SURFELWARP_CHECK_EQ(reference_vertex.Size(), knn.Size());
	SURFELWARP_CHECK_EQ(knn.Size(), knn_weight.Size());
	SURFELWARP_CHECK_EQ(knn.Size(), live_vertex.Size());
	SURFELWARP_CHECK_EQ(node_se3.Size(), m_node_se3.DeviceArraySize());

	dim3 blk(256);
	dim3 grid(divUp(reference_vertex.Size(), blk.x));
	device::forwardWarpVertexDebugKernel<<<grid, blk>>>(
		reference_vertex,
		knn.RawPtr(),
		knn_weight.RawPtr(), 
		node_se3.RawPtr(), 
		live_vertex.RawPtr()
	);
}
