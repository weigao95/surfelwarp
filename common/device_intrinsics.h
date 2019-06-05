//
// Created by wei on 2/14/18.
//

#pragma once

#include <vector_functions.h>

namespace surfelwarp {

    __device__ __forceinline__
    float shfl_add(float x, int offset)
    {
        float result = 0;

        asm volatile (
        "{.reg .f32 r0;"
                ".reg .pred p;"
                "shfl.up.b32 r0|p, %1, %2, 0;"
                "@p add.f32 r0, r0, %1;"
                "mov.f32 %0, r0;}"
        : "=f"(result) : "f"(x), "r"(offset));

        return result;
    }

    __device__ __forceinline__
    float warp_scan(float data)
    {
        data = shfl_add(data, 1);
        data = shfl_add(data, 2);
        data = shfl_add(data, 4);
        data = shfl_add(data, 8);
        data = shfl_add(data, 16);
        return data;
    }
    
    
    __device__ __forceinline__
    int shfl_add(int x, int offset)
    {
        int result = 0;
        
        asm volatile (
        "{.reg .s32 r0;"
            ".reg .pred p;"
            "shfl.up.b32 r0|p, %1, %2, 0;"
            "@p add.s32 r0, r0, %1;"
            "mov.s32 %0, r0;}"
        : "=r"(result) : "r"(x), "r"(offset));
        
        return result;
    }
    
    __device__ __forceinline__
    int warp_scan(int data)
    {
        data = shfl_add(data, 1);
        data = shfl_add(data, 2);
        data = shfl_add(data, 4);
        data = shfl_add(data, 8);
        data = shfl_add(data, 16);
        return data;
    }

	__device__ __forceinline__
    unsigned shfl_add(unsigned x, unsigned offset)
    {
        unsigned result = 0;
        
        asm volatile (
        "{.reg .u32 r0;"
            ".reg .pred p;"
            "shfl.up.b32 r0|p, %1, %2, 0;"
            "@p add.u32 r0, r0, %1;"
            "mov.u32 %0, r0;}"
        : "=r"(result) : "r"(x), "r"(offset));
        
        return result;
    }

	__device__ __forceinline__
    unsigned warp_scan(unsigned data)
    {
        data = shfl_add(data, 1u);
        data = shfl_add(data, 2u);
        data = shfl_add(data, 4u);
        data = shfl_add(data, 8u);
        data = shfl_add(data, 16u);
        return data;
    }
}
