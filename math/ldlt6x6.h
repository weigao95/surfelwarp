//
// Created by wei on 2/8/18.
//

#pragma once

#include "common/common_types.h"

namespace surfelwarp {

    //An class to perform ldlt decomposition on 6x6 symmetric positve-definite matrix
    struct ldlt6x6 {
        //Store the input in matrix, the output in factored, factored are assumed to has 36 elements
        //While only the top-right triangle part is used
        __host__ __device__ static void factor(const float *matrix, float *factored);

        //The flattened position in *factored: top left part
        __host__ __device__ __forceinline__ static int flatten(int i, int j) {
            if (i < j) {
                return 6 * i + j;
            } else {
                return 6 * j + i;
            }
        }

        //The flatten of different patterns
        __host__ __device__ __forceinline__ static int topright_flatten(int i, int j) {
            if (i < j) {
                return 6 * i + j;
            } else {
                return 6 * j + i;
            }
        }
        __host__ __device__ __forceinline__ static int bottomleft_flatten(int i, int j) {
            if (i > j) {
                return 6 * i + j;
            } else {
                return 6 * j + i;
            }
        }

    };
}
