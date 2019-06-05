#pragma once
#include <vector_types.h>

namespace surfelwarp {

	/* The struct for the range of numeric values
	*/
	template <typename T> struct numeric_limits;


	/* For float type, the host and device code might be different
	*/
	template <> struct numeric_limits<float> {
#if defined (__CUDACC__)
		__device__ __forceinline__ static float quiet_nan() {
			return __int_as_float(0x7fffffff);
		}
#else
		__host__ __device__ __forceinline__ static float quiet_nan() {
			int value = 0x7fffffff;
			return (*((float*)&(value)));
		}
#endif

		__host__ __device__ __forceinline__ static float epsilon() {
			return 1.192092896e-07f;
		}

		__host__ __device__ __forceinline__ static float min_positive() {
			return 1.175494351e-38f;
		}

		__host__ __device__ __forceinline__ static float max() {
			return 3.402823466e+38f;
		}
	};

	/* For 16 bit signed short
	*/
	template <> struct numeric_limits<signed short> {
		__host__ __device__ __forceinline__ static signed short min() {
			return -32768;
		}

		__host__ __device__ __forceinline__ static signed short max() {
			return 32767;
		}
	};

	/* For 16 bit unsigned short type
	*/
	template <> struct numeric_limits<unsigned short> {
		__host__ __device__ __forceinline__ static unsigned short min() {
			return 0;
		}

		__host__ __device__ __forceinline__ static unsigned short max() {
			return 0xffff;
		}
	};
}