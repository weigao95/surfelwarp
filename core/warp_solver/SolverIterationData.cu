#include "core/warp_solver/SolverIterationData.h"
#include "core/warp_solver/solver_types.h"
#include "math/vector_ops.hpp"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__global__ void applyWarpFieldUpdateKernel(
		const DeviceArrayView<DualQuaternion> warp_field,
		const float* _warp_field_update,
		DualQuaternion* updated_warpfield,
		const float coef
	) {
		const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
		if (tidx >= warp_field.Size()) return;
		float3 twist_rot;
		twist_rot.x = coef * _warp_field_update[6 * tidx];
		twist_rot.y = coef * _warp_field_update[6 * tidx + 1];
		twist_rot.z = coef * _warp_field_update[6 * tidx + 2];

		float3 twist_trans;
		twist_trans.x = coef * _warp_field_update[6 * tidx + 3];
		twist_trans.y = coef * _warp_field_update[6 * tidx + 4];
		twist_trans.z = coef * _warp_field_update[6 * tidx + 5];

		mat34 SE3;
		if (fabsf_sum(twist_rot) < 1e-4f) {
			SE3.rot = mat33::identity();
		}
		else {
			const float angle = norm(twist_rot);
			const float3 axis = 1.0f / angle * twist_rot;

			float c = cosf(angle);
			float s = sinf(angle);
			float t = 1.0f - c;

			SE3.rot.m00() = t*axis.x*axis.x + c;
			SE3.rot.m01() = t*axis.x*axis.y - axis.z*s;
			SE3.rot.m02() = t*axis.x*axis.z + axis.y*s;

			SE3.rot.m10() = t*axis.x*axis.y + axis.z*s;
			SE3.rot.m11() = t*axis.y*axis.y + c;
			SE3.rot.m12() = t*axis.y*axis.z - axis.x*s;

			SE3.rot.m20() = t*axis.x*axis.z - axis.y*s;
			SE3.rot.m21() = t*axis.y*axis.z + axis.x*s;
			SE3.rot.m22() = t*axis.z*axis.z + c;
		}

		SE3.trans = twist_trans;

		mat34 SE3_prev = warp_field[tidx];
		SE3_prev = SE3 * SE3_prev;
		updated_warpfield[tidx] = SE3_prev;
	}

} // namespace device
} // namespace surfelwarp


void surfelwarp::SolverIterationData::ApplyWarpFieldUpdate(cudaStream_t stream, float step) {
	//Determine which node list updated to
	const auto init_dq = CurrentWarpFieldInput();
	DeviceArraySlice<DualQuaternion> updated_dq;
	switch (m_updated_se3) {
	case IterationInputFrom::WarpFieldInit:
	case IterationInputFrom::SE3_Buffer_1:
		updated_dq = node_se3_0_.ArraySlice();
		break;
	case IterationInputFrom::SE3_Buffer_0:
		updated_dq = node_se3_1_.ArraySlice();
		break;
	}

	//Invoke the kernel
	dim3 blk(64);
	dim3 grid(divUp(NumNodes(), blk.x));
	device::applyWarpFieldUpdateKernel<<<grid, blk, 0, stream>>>(
		init_dq,
		m_twist_update.Ptr(),
		updated_dq.RawPtr(),
		step
	);

	//Update the flag
	updateIterationFlags();

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}