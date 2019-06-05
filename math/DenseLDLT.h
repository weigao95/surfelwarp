//
// Created by wei on 2/8/18.
//

#pragma once

#include "common/common_types.h"

namespace surfelwarp {

    template <int N>
    struct DenseLDLT {
        //The flatten position of different patterns
        __host__ __device__ __forceinline__ static int toprightFlatten(int i, int j) {
            if (i < j) {
                return N * i + j;
            } else {
                return N * j + i;
            }
        }
        __host__ __device__ __forceinline__ static int bottomleftFlatten(int i, int j) {
            if (i > j) {
                return N * i + j;
            } else {
                return N * j + i;
            }
        }

        //Do factor with additional space
        __host__ __device__ static void Factor(float* matrix, float* factored);
		//Do factor inplace
        __host__ __device__ static void Factor(float* matrix);

		//Solve Ax=b given a factored matrix
		__host__ __device__ static void Solve(const float* factored, float* b, float* auxiliary);
		__host__ __device__ static void Solve(const float* factored, float* b);
    };
}


#include "math/DenseLDLT.hpp"