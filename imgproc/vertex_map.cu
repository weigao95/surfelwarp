#include "common/global_configs.h"
#include "imgproc/generate_maps.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	//Compute the confidence value of a depth pixel
	//Use 1.0f at first
	__host__ __device__ __forceinline__
		float confidence_value(const unsigned short depth_value, const float view_angle_dot = 1.0f) {
		return 1.0f;
	}

	__global__ void createVertexConfidMapKernel(
		cudaTextureObject_t depth_img,
		const unsigned rows, const unsigned cols,
		const IntrinsicInverse intrinsic_inv,
		cudaSurfaceObject_t vertex_confid_map
	) {
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x >= cols || y >= rows) return;

		//Obtain the value and perform back-projecting
		const unsigned short raw_depth = tex2D<unsigned short>(depth_img, x, y);
		float4 vertex_confid;

		//scale the depth to [m]
		//The depth image is always in [mm]
		vertex_confid.z = float(raw_depth) / (1000.f);
		vertex_confid.x = (x - intrinsic_inv.principal_x) * intrinsic_inv.inv_focal_x * vertex_confid.z;
		vertex_confid.y = (y - intrinsic_inv.principal_y) * intrinsic_inv.inv_focal_y * vertex_confid.z;
		vertex_confid.w = confidence_value(raw_depth);
		surf2Dwrite(vertex_confid, vertex_confid_map, x * sizeof(float4), y);
	}

}; /* End of namespace device */
}; /* End of namespace surfelwarp */



void surfelwarp::createVertexConfigMap(
	cudaTextureObject_t depth_img, 
	const unsigned rows, const unsigned cols, 
	const IntrinsicInverse intrinsic_inv, 
	cudaSurfaceObject_t vertex_confid_map,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(cols, blk.x), divUp(rows, blk.y));
	device::createVertexConfidMapKernel<<<grid, blk, 0, stream>>>(
		depth_img, 
		rows, cols, 
		intrinsic_inv,
		vertex_confid_map
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}