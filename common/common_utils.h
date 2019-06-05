//
// Created by wei on 2/16/18.
//

#pragma once

#include "common/common_types.h"
#include <cuda_texture_types.h>
#include <exception>
#include <cuda.h>

namespace surfelwarp {

    /**
     * \brief The general swap operation on device.
     */
    template <typename T>
    __host__ __device__ __forceinline__ void swap(T& a, T& b) noexcept
    {
        T c(a); a = b; b = c;
    }

    //Texture fetching for 1D array: These names are supposed to be more clear
#if defined(__CUDA_ARCH__)
    template<typename T>
    __device__ __forceinline__ T fetch1DLinear(cudaTextureObject_t texObj, int x) {
        return tex1Dfetch<T>(texObj, x);
    }

    template<typename T>
    __device__ __forceinline__ T fetch1DArray(cudaTextureObject_t texObj, float x) {
        return tex1D<T>(texObj, x);
    }
#else
    template<typename T>
    __host__ __forceinline__ T fetch1DLinear(cudaTextureObject_t texObj, int x) {
        throw new std::runtime_error("The texture object can only be accessed on device! ");
    }

    template<typename T>
    __host__ __forceinline__ T fetch1DArray(cudaTextureObject_t texObj, float x) {
        throw new std::runtime_error("The texture object can only be accessed on device! ");
    }
#endif

    

	/**
	* \brief The inverse of intrisic
	*/
	__host__ __forceinline__ IntrinsicInverse inverse(const Intrinsic& intrinsic) {
		IntrinsicInverse intr_inv;
		intr_inv.principal_x = intrinsic.principal_x;
		intr_inv.principal_y = intrinsic.principal_y;
		intr_inv.inv_focal_x = 1.0f / double(intrinsic.focal_x);
		intr_inv.inv_focal_y = 1.0f / double(intrinsic.focal_y);
		return intr_inv;
    }

    /**
     * \brief Initialize the context and driver api for cuda operations
     */
    CUcontext initCudaContext(int selected_device = 0);


    /**
     * \brief Clear the cuda context at the end of program
     */
    void destroyCudaContext(CUcontext context);
}
