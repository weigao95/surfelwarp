#pragma once
#include "common/common_utils.h"
#include "pcg_solver/solver_configs.h"
#include "pcg_solver/BinBlockCSR.h"

template<int BlockSize>
float surfelwarp::BinBlockCSR<BlockSize>::SparseMV(
	const float * A_data,
	const int * A_colptr,
	const int * A_rowptr,
	const float * x,
	const int row
) {
	auto begin = A_rowptr[row];
	const auto end = A_rowptr[row + bin_size];
	const auto inbin_offset = begin & (bin_size - 1);
	int column_offset = ((begin - inbin_offset) / BlockSize) + inbin_offset;
	float sp_mv = 0.0f;
	while (begin < end)
	{
		const auto column = A_colptr[column_offset];
		for (auto i = 0; i < BlockSize; i++)
		{
			const float matrix_data = A_data[begin];
			const float x_data = (column >= 0 ? x[column + i] : 0.0f);
			sp_mv += (matrix_data * x_data);
			begin += bin_size;
		}
		//Increase the column offset
		column_offset += bin_size;
	}

	return sp_mv;
}

template<int BlockSize>
__device__ float surfelwarp::BinBlockCSR<BlockSize>::SparseMV(
		const float *A_data,
		const int *A_colptr,
		const int *A_rowptr,
		cudaTextureObject_t x,
		const int row
) {
	auto begin = A_rowptr[row];
	const auto end = A_rowptr[row + bin_size];
	const auto inbin_offset = begin & (bin_size - 1);
	int column_offset = ((begin - inbin_offset) / BlockSize) + inbin_offset;
	float sp_mv = 0.0f;
	while (begin < end)
	{
		const auto column = A_colptr[column_offset];
		for (auto i = 0; i < BlockSize; i++)
		{
			const float matrix_data = A_data[begin];
			float x_data = 0.0f;
			if(column >= 0)
				x_data = fetch1DLinear<float>(x, column + i);
			sp_mv += (matrix_data * x_data);
			begin += bin_size;
		}
		//Increase the column offset
		column_offset += bin_size;
	}

	return sp_mv;
}
