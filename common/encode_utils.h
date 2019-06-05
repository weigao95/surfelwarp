#pragma once

#include "common/common_types.h"
#include <cuda_texture_types.h>
#include <exception>
#include <cuda.h>

namespace surfelwarp
{
	
	//The transfer between uint and float
#if defined(__CUDA_ARCH__)
	__device__ __forceinline__ float uint_as_float(unsigned i_value) {
		return __uint_as_float(i_value);
	}

	__device__ __forceinline__ unsigned float_as_uint(float f_value) {
		return __float_as_uint(f_value);
	}
#else
	//The transfer from into to float
	union uint2float_union {
		unsigned i_value;
		float f_value;
	};

	//The methods
	__host__ __forceinline__ float uint_as_float(unsigned i_value) {
		uint2float_union u;
		u.i_value = i_value;
		return u.f_value;
	}
	__host__ __forceinline__ unsigned float_as_uint(float f_value) {
		uint2float_union u;
		u.f_value = f_value;
		return u.i_value;
	}
#endif

	/**
	* \brief Encode uint8_t as float32_t or uint32_t
	*/
	__host__ __device__ __forceinline__ unsigned uint_encode_rgba(
		const unsigned char r,
		const unsigned char g,
		const unsigned char b,
		const unsigned char a
	) {
		const unsigned encoded = ((a << 24) + (r << 16) + (g << 8) + b);
		return encoded;
	}

	__host__ __device__ __forceinline__ unsigned uint_encode_rgb(
		const unsigned char r,
		const unsigned char g,
		const unsigned char b
	) {
		const unsigned encoded = ((r << 16) + (g << 8) + b);
		return encoded;
	}

	__host__ __device__ __forceinline__ 
	unsigned uint_encode_rgb(const uchar3 rgb) {
		const unsigned encoded = ((rgb.x << 16) + (rgb.y << 8) + rgb.z);
		return encoded;
	}

	__host__ __device__ __forceinline__ float float_encode_rgba(
		const unsigned char r,
		const unsigned char g,
		const unsigned char b,
		const unsigned char a
	) {
		return uint_as_float(uint_encode_rgba(r, g, b, a));
	}

	__host__ __device__ __forceinline__ 
	float float_encode_rgb(const uchar3 rgb) {
		return uint_as_float(uint_encode_rgb(rgb));
	}

	/**
	* \brief Dncode uint8_t as float32_t or uint32_t
	*/
	__host__ __device__ __forceinline__ void uint_decode_rgba(
		const unsigned encoded,
		unsigned char& r,
		unsigned char& g,
		unsigned char& b,
		unsigned char& a
	) {
		a = ((encoded & 0xff000000) >> 24);
		r = ((encoded & 0x00ff0000) >> 16);
		g = ((encoded & 0x0000ff00) >> 8);
		b = ((encoded & 0x000000ff) /*0*/);
	}

	__host__ __device__ __forceinline__ void uint_decode_rgb(
		const unsigned encoded,
		unsigned char& r,
		unsigned char& g,
		unsigned char& b
	){
		r = ((encoded & 0x00ff0000) >> 16);
		g = ((encoded & 0x0000ff00) >> 8);
		b = ((encoded & 0x000000ff) /*0*/);
	}

	__host__ __device__ __forceinline__ void uint_decode_rgb(
		const unsigned encoded,
		uchar3& rgb
	) {
		uint_decode_rgb(encoded, rgb.x, rgb.y, rgb.z);
	}

	__host__ __device__ __forceinline__ void float_decode_rgba(
		const float encoded,
		unsigned char& r,
		unsigned char& g,
		unsigned char& b,
		unsigned char& a
	) {
		const auto unsigned_encoded = float_as_uint(encoded);
		uint_decode_rgba(unsigned_encoded, r, g, b, a);
	}

	__host__ __device__ __forceinline__ 
	void float_decode_rgb(const float encoded, uchar3& rgb)
	{
		const auto unsigned_encoded = float_as_uint(encoded);
		uint_decode_rgb(unsigned_encoded, rgb);
	}

	//Assume x, y, z are in (-512, 512)
	__host__ __device__ __forceinline__ 
	int encodeVoxel(const int x, const int y, const int z) {
		return (x + 512) + (y + 512) * 1024 + (z + 512) * 1024 * 1024;
	}

	__host__ __device__ __forceinline__ 
	void decodeVoxel(const int encoded, int& x, int& y, int& z) {
		z = encoded / (1024 * 1024);
		x = encoded % 1024;
		y = (encoded - z * 1024 * 1024) / 1024;
		x -= 512;
		y -= 512;
		z -= 512;
	}
}