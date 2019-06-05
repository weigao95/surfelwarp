#include "core/SurfelGeometry.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__global__ void applySE3DebugKernel(
		const mat34 se3,
		DeviceArraySlice<float4> referece_vertex_confid,
		float4* reference_normal_radius,
		float4* live_vertex_confid,
		float4* live_normal_radius
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < referece_vertex_confid.Size()) {
			float4 ref_v4 = referece_vertex_confid[idx];
			float3 transformed_ref_v3 = se3.rot * ref_v4 + se3.trans;
			referece_vertex_confid[idx] = make_float4(transformed_ref_v3.x, transformed_ref_v3.y, transformed_ref_v3.z, ref_v4.w);

			float4 ref_n4 = reference_normal_radius[idx];
			float3 transformed_ref_n3 = se3.rot * ref_n4;
			reference_normal_radius[idx] = make_float4(transformed_ref_n3.x, transformed_ref_n3.y, transformed_ref_n3.z, ref_n4.w);

			float4 live_v4 = live_vertex_confid[idx];
			float3 transformed_live_v3 = se3.rot * live_v4 + se3.trans;
			live_vertex_confid[idx] = make_float4(transformed_live_v3.x, transformed_live_v3.y, transformed_live_v3.z, live_v4.w);

			float4 live_n4 = live_normal_radius[idx];
			float3 transformed_live_n3 = se3.rot * live_n4;
			live_normal_radius[idx] = make_float4(transformed_live_n3.x, transformed_live_n3.y, transformed_live_n3.z, live_n4.w);
		}
	}


} // device 
} // surfelwarp



void surfelwarp::SurfelGeometry::AddSE3ToVertexAndNormalDebug(const mat34 & se3) {
	dim3 blk(128);
	dim3 grid(divUp(NumValidSurfels(), blk.x));
	device::applySE3DebugKernel<<<grid, blk>>>(
		se3, 
		m_reference_vertex_confid.ArraySlice(), 
		m_reference_normal_radius.Ptr(),
		m_live_vertex_confid.Ptr(),
		m_live_normal_radius.Ptr()
	);

	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
}