#pragma once

#include "common/common_utils.h"

template<int N>
void surfelwarp::DenseGaussian<N>::Inverse(float * matrix)
{
	//Main loop
	for (auto i = 0; i < N; i++)
	{
		//Scale the current row
		float inv_aii = 1.0f / matrix[flatten(i, i)];
		for (auto k = 0; k < N; k++) {
			matrix[flatten(i, k)] *= inv_aii;
		}
		matrix[flatten(i, i)] = inv_aii;

		//Eliminate rows below and above current row
		for (auto r = 0; r < N; r++) {
			if (r == i) continue;
			float a_ri = matrix[flatten(r, i)];
			for (auto k = 0; k < N; k++) {
				matrix[flatten(r, k)] -= a_ri * matrix[flatten(i, k)];
			}
			matrix[flatten(r, i)] = -a_ri * matrix[flatten(i, i)];
		}
	}
}


template<int N>
void surfelwarp::DenseGaussian<N>::PivotInverse(float * matrix, int * relocated_cols)
{
	//Init of permutation
	int permutation[N];
	for (auto i = 0; i < N; i++) permutation[i] = i;

	//Main loop
	for (auto i = 0; i < N; i++)
	{
		//Init the pivot
		int pivot = i;
		float pivot_value = fabsf(matrix[flatten(i, i)]);

		//Search the pivot
		for (auto j = i + 1; j < N; j++) {
			if (pivot_value < fabsf(matrix[flatten(j, i)])) {
				pivot_value = fabsf(matrix[flatten(j, i)]);
				pivot = j;
			}
		}

		//Swap the pivot row with the current row
		for (auto k = 0; k < N; k++) {
			swap(matrix[flatten(i, k)], matrix[flatten(pivot, k)]);
		}

		//Swap the permutation vector
		swap(permutation[i], permutation[pivot]);
		relocated_cols[i] = permutation[i];

		//Scale the current row
		float inv_aii = 1.0f / matrix[flatten(i, i)];
		for (auto k = 0; k < N; k++) {
			matrix[flatten(i, k)] *= inv_aii;
		}
		matrix[flatten(i, i)] = inv_aii;

		//Eliminate rows below and above current row
		for (auto r = 0; r < N; r++) {
			if (r != i) {
                float a_ri = matrix[flatten(r, i)];
                for (auto k = 0; k < N; k++) {
                    matrix[flatten(r, k)] -= a_ri * matrix[flatten(i, k)];
                }
                matrix[flatten(r, i)] = -a_ri * matrix[flatten(i, i)];
            }
		}
	}
}

template<int N>
void surfelwarp::DenseGaussian<N>::BuildInversed(
	const float * raw_inversed, 
	const int * relocated_cols, 
	float * inversed
) {
	for (auto j = 0; j < N; j++) {
		for (auto i = 0; i < N; i++) {
			inversed[N * relocated_cols[j] + i] = raw_inversed[flatten(i, j)];
		}
	}
}

template<int N>
__host__ __device__ void surfelwarp::DenseGaussian<N>::Inverse(float * matrix, float * inversed)
{
	int relocated_cols[N];
	PivotInverse(matrix, relocated_cols);
	BuildInversed(matrix, relocated_cols, inversed);
}