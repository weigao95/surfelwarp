#pragma once
#include <vector_functions.h>

namespace surfelwarp {
	
	/**
	 * \brief Format convert
	 * \param rgba in the scale of [0, 1]
	 * \param ycrcb in the scale of [0, 1]
	 */
	__host__ __device__ __forceinline__
	void normalized_rgba2ycrcb(const float4& rgba, float3& ycrcb) {
		ycrcb.x =     0.299f * rgba.x +    0.587f * rgba.y + 0.114f * rgba.z;
		ycrcb.y = -0.168736f * rgba.x - 0.331264f * rgba.y + 0.5f * rgba.z;
		ycrcb.z =       0.5f * rgba.x - 0.418688f * rgba.y - 0.091312f * rgba.z;
	}

	__host__ __device__ __forceinline__
	void normalized_rgb2ycrcb(const float3& rgb, float3& ycrcb) {
		ycrcb.x =     0.299f * rgb.x +    0.587f * rgb.y + 0.114f * rgb.z;
		ycrcb.y = -0.168736f * rgb.x - 0.331264f * rgb.y + 0.5f * rgb.z;
		ycrcb.z =       0.5f * rgb.x - 0.418688f * rgb.y - 0.091312f * rgb.z;
	}
	
	__host__ __device__ __forceinline__
	float rgba2density(const float4& rgba) {
		return 0.299f * rgba.x +    0.587f * rgba.y + 0.114f * rgba.z;
	}
	
	__host__ __device__ __forceinline__
	float rgb2density(const float3& rgb) {
		return 0.299f * rgb.x +    0.587f * rgb.y + 0.114f * rgb.z;
	}

	__host__ __device__ __forceinline__
	float rgb2density(const float4& rgb) {
		return 0.299f * rgb.x +    0.587f * rgb.y + 0.114f * rgb.z;
	}

	__host__ __device__ __forceinline__
	unsigned char rgb_diff_abs(const uchar3& rgb_0, const uchar3& rgb_1) {
		return abs(rgb_0.x - rgb_1.x) + abs(rgb_0.y - rgb_1.y) + abs(rgb_0.z - rgb_1.z);
	}
}