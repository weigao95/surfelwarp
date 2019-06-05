#pragma once
#include "common/common_types.h"

namespace surfelwarp {

	template<int N>
	struct DenseGaussian {
		//The matrix is assumed to be colume major
		__host__ __device__ __forceinline__ static int flatten(int row, int col) {
			return row + col * N;
		}

		//Do elimination inplace
		__host__ __device__ static void Inverse(float* matrix);

		//Do elimination with pivoting
		//matrix: float[N * N], colume major
		//relocated cols: int[N]
		__host__ __device__ static void PivotInverse(float* matrix, int* relocated_cols);
		__host__ __device__ static void Inverse(float* matrix, float* inversed);

		//Reconstruct the matrix given the
		//inversed result and relocated array
		__host__ __device__ static void BuildInversed(const float* raw_inversed, const int* relocated_cols, float* inversed);
	};
}

#include "math/DenseGaussian.hpp"