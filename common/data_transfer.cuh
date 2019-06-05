#pragma once

#include "common/common_types.h"
#include "common/common_utils.h"
#include "common/data_transfer.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {


	template<typename T>
	__global__ void textureToMap2DKernel(
		cudaTextureObject_t texture,
		PtrStepSz<T> map
	) {
		const auto x = threadIdx.x + blockDim.x*blockIdx.x;
		const auto y = threadIdx.y + blockDim.y*blockIdx.y;
		if (x >= map.cols || y >= map.rows) return;
		T element = tex2D<T>(texture, x, y);
		map.ptr(y)[x] = element;
	}

}; /* End of namespace device */ }; /* End of namespace surfelwarp */



template<typename T>
void surfelwarp::textureToMap2D(
	cudaTextureObject_t texture,
	DeviceArray2D<T>& map,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(map.cols(), blk.x), divUp(map.rows(), blk.y));
	device::textureToMap2DKernel<T><<<grid, blk, 0, stream>>>(texture, map);
}












