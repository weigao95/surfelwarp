#pragma once

#include "common/common_types.h"
#include "math/device_mat.h"
#include "math/numeric_limits.hpp"
#include "math/vector_ops.hpp"

namespace surfelwarp {


	/* The struct contains method to compute the eigen vector
	*  of 3x3 positive definite matrix, for normal computation
	*  Copy and modify from the pcl implementation
	*/
	struct eigen33 {
	private:
		template<int Rows>
		struct MiniMat
		{
			float3 data[Rows];
			__host__ __device__ __forceinline__ float3& operator[](int i) { return data[i]; }
			__host__ __device__ __forceinline__ const float3& operator[](int i) const { return data[i]; }
		};
		typedef MiniMat<3> Mat33;

	public:
		//Compute the root of x^2 - b x + c = 0, assuming real roots
		__host__ __device__ static void compute_root2(const float b, const float c, float3& roots);

		//Compute the root of x^3 - c2*x^2 + c1*x - c0 = 0
		__host__ __device__ static void compute_root3(const float c0, const float c1, const float c2, float3& roots);

		//Constructor
		__host__ __device__ eigen33(float* psd33) : psd_matrix33(psd33) {}

		//Compute the eigen vectors
		__host__ __device__ static float3 unit_orthogonal(const float3& src);
		__host__ __device__ __forceinline__ void compute(Mat33& tmp, Mat33& vec_tmp, Mat33& evecs, float3& evals);
		__host__ __device__ __forceinline__ void compute(float3& eigen_vec);
		__host__ __device__ __forceinline__ void compute(float3& eigen_vec, float& eigen_value);

	private:
		//The psd matrix to compute eigen vector, in the size of 6
		float* psd_matrix33;

		//For accessing
		__host__ __device__  __forceinline__ float m00() const { return psd_matrix33[0]; }
		__host__ __device__  __forceinline__ float m01() const { return psd_matrix33[1]; }
		__host__ __device__  __forceinline__ float m02() const { return psd_matrix33[2]; }
		__host__ __device__  __forceinline__ float m10() const { return psd_matrix33[1]; }
		__host__ __device__  __forceinline__ float m11() const { return psd_matrix33[3]; }
		__host__ __device__  __forceinline__ float m12() const { return psd_matrix33[4]; }
		__host__ __device__  __forceinline__ float m20() const { return psd_matrix33[2]; }
		__host__ __device__  __forceinline__ float m21() const { return psd_matrix33[4]; }
		__host__ __device__  __forceinline__ float m22() const { return psd_matrix33[5]; }

		__host__ __device__  __forceinline__ float3 row0() const { return make_float3(m00(), m01(), m02()); }
		__host__ __device__  __forceinline__ float3 row1() const { return make_float3(m10(), m11(), m12()); }
		__host__ __device__  __forceinline__ float3 row2() const { return make_float3(m20(), m21(), m22()); }

		__host__ __device__ __forceinline__ static bool isMuchSmallerThan(float x, float y) {
			const float prec_sqr = numeric_limits<float>::epsilon() * numeric_limits<float>::epsilon();
			return x * x <= prec_sqr * y * y;
		}

		//The inverse sqrt function
#if defined (__CUDACC__)
		__host__ __device__ __forceinline__ static float inv_sqrt(float x) {
			return rsqrtf(x);
		}
#else
		__host__ __device__ __forceinline__ static float inv_sqrt(float x) {
			return 1.0f / sqrtf(x);
		}
#endif
	};
}


#include "math/eigen33.hpp"