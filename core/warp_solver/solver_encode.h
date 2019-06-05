#pragma once
#include <vector_types.h>

namespace surfelwarp
{
	//To encode and decode the pair into int, this number shall
	//be larger than any of the y (which is the node index)
	const int large_number = 4096;

	//A row major encoding of the (row, col) pair
	__host__ __device__ __forceinline__ unsigned encode_nodepair(unsigned short x, unsigned short y) {
		return x * large_number + y;
	}
	__host__ __device__ __forceinline__ void decode_nodepair(const unsigned encoded, unsigned& x, unsigned& y) {
		x = encoded / large_number;
		y = encoded % large_number;
	}

	__host__ __device__ __forceinline__ unsigned short encoded_row(const unsigned encoded) {
		return encoded / large_number;
	}

	__host__ __device__ __forceinline__ unsigned short encoded_col(const unsigned encoded) {
		return encoded % large_number;
	}
}
