#pragma once
#include "common/common_utils.h"
#include "math/eigen33.h"
#include "math/vector_ops.hpp"

__host__ __device__ __forceinline__ void surfelwarp::eigen33::compute_root2(const float b, const float c, float3 & roots)
{
	roots.x = 0.0f; //Used in compute_root3
	float d = b * b - 4.f * c;
	if (d < 0.f) // no real roots!!!! THIS SHOULD NOT HAPPEN!
		d = 0.f;

	float sd = sqrtf(d);

	roots.z = 0.5f * (b + sd);
	roots.y = 0.5f * (b - sd);
}


__host__ __device__ __forceinline__ float3 surfelwarp::eigen33::unit_orthogonal(const float3 & src)
{
	float3 perp;
	/* Compute the crossed product of *this with a vector
	*  that is not too close to being colinear to *this.
	*/

	/* unless the x and y coords are both close to zero, we can
	* simply take ( -y, x, 0 ) and normalize it.
	*/
	if (!isMuchSmallerThan(src.x, src.z) || !isMuchSmallerThan(src.y, src.z))
	{
		float invnm = inv_sqrt(src.x*src.x + src.y*src.y);
		perp.x = -src.y * invnm;
		perp.y = src.x * invnm;
		perp.z = 0.0f;
	}
	/* if both x and y are close to zero, then the vector is close
	* to the z-axis, so it's far from colinear to the x-axis for instance.
	* So we take the crossed product with (1,0,0) and normalize it.
	*/
	else
	{
		float invnm = inv_sqrt(src.z * src.z + src.y * src.y);
		perp.x = 0.0f;
		perp.y = -src.z * invnm;
		perp.z = src.y * invnm;
	}

	return perp;
}



__host__ __device__ __forceinline__ void surfelwarp::eigen33::compute_root3(const float c0, const float c1, const float c2, float3 & roots)
{
	if (fabsf(c0) < numeric_limits<float>::epsilon()) {
		compute_root2(c2, c1, roots);
	}
	else {
		const float s_inv3 = 1.f / 3.f;
		const float s_sqrt3 = sqrtf(3.f);
		// Construct the parameters used in classifying the roots of the equation
		// and in solving the equation for the roots in closed form.
		float c2_over_3 = c2 * s_inv3;
		float a_over_3 = (c1 - c2 * c2_over_3) * s_inv3;
		if (a_over_3 > 0.f)
			a_over_3 = 0.f;
		float half_b = 0.5f * (c0 + c2_over_3 * (2.f * c2_over_3 * c2_over_3 - c1));
		float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
		if (q > 0.f)
			q = 0.f;

		//Compute the eigenvalues by solving for the roots of the polynomial
		float rho = sqrtf(-a_over_3);
		float theta = atan2f(sqrtf(-q), half_b)*s_inv3;

		//Using intrinsic here
		float cos_theta, sin_theta;
		cos_theta = cosf(theta);
		sin_theta = sinf(theta);
		//Compute the roots
		roots.x = c2_over_3 + 2.f * rho * cos_theta;
		roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
		roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

		//Sort the root according to their values
		if (roots.x >= roots.y)
			swap(roots.x, roots.y);

		if (roots.y >= roots.z) {
			swap(roots.y, roots.z);
			if (roots.x >= roots.y)
				swap(roots.x, roots.y);
		}

		//eigenvalues for symetric positive semi-definite matrix can not be negative! Set it to 0
		if (roots.x <= 0.0f)
			compute_root2(c2, c1, roots);
	}
}

__host__ __device__ __forceinline__ void surfelwarp::eigen33::compute(Mat33 & tmp, Mat33 & vec_tmp, Mat33 & evecs, float3 & evals)
{
	// Scale the matrix so its entries are in [-1,1].  The scaling is applied
	// only when at least one matrix entry has magnitude larger than 1.
	float max01 = fmaxf(fabsf(psd_matrix33[0]), fabsf(psd_matrix33[1]));
	float max23 = fmaxf(fabsf(psd_matrix33[2]), fabsf(psd_matrix33[3]));
	float max45 = fmaxf(fabsf(psd_matrix33[4]), fabsf(psd_matrix33[5]));
	float m0123 = fmaxf(max01, max23);
	float scale = fmaxf(max45, m0123);

	if (scale <= numeric_limits<float>::min_positive())
		scale = 1.f;

	psd_matrix33[0] /= scale;
	psd_matrix33[1] /= scale;
	psd_matrix33[2] /= scale;
	psd_matrix33[3] /= scale;
	psd_matrix33[4] /= scale;
	psd_matrix33[5] /= scale;

	// The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
	// eigenvalues are the roots to this equation, all guaranteed to be
	// real-valued, because the matrix is symmetric.
	float c0 = m00() * m11() * m22()
		+ 2.f * m01() * m02() * m12()
		- m00() * m12() * m12()
		- m11() * m02() * m02()
		- m22() * m01() * m01();
	float c1 = m00() * m11() -
		m01() * m01() +
		m00() * m22() -
		m02() * m02() +
		m11() * m22() -
		m12() * m12();
	float c2 = m00() + m11() + m22();

	compute_root3(c0, c1, c2, evals);

	if (evals.z - evals.x <= numeric_limits<float>::epsilon())
	{
		evecs[0] = make_float3(1.f, 0.f, 0.f);
		evecs[1] = make_float3(0.f, 1.f, 0.f);
		evecs[2] = make_float3(0.f, 0.f, 1.f);
	}
	else if (evals.y - evals.x <= numeric_limits<float>::epsilon())
	{
		// first and second equal                
		tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
		tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

		vec_tmp[0] = cross(tmp[0], tmp[1]);
		vec_tmp[1] = cross(tmp[0], tmp[2]);
		vec_tmp[2] = cross(tmp[1], tmp[2]);

		float len1 = dot(vec_tmp[0], vec_tmp[0]);
		float len2 = dot(vec_tmp[1], vec_tmp[1]);
		float len3 = dot(vec_tmp[2], vec_tmp[2]);

		if (len1 >= len2 && len1 >= len3)
		{
			evecs[2] = vec_tmp[0] * inv_sqrt(len1);
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			evecs[2] = vec_tmp[1] * inv_sqrt(len2);
		}
		else
		{
			evecs[2] = vec_tmp[2] * inv_sqrt(len3);
		}

		evecs[1] = unit_orthogonal(evecs[2]);
		evecs[0] = cross(evecs[1], evecs[2]);
	}
	else if (evals.z - evals.y <= numeric_limits<float>::epsilon())
	{
		// second and third equal                                    
		tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
		tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

		vec_tmp[0] = cross(tmp[0], tmp[1]);
		vec_tmp[1] = cross(tmp[0], tmp[2]);
		vec_tmp[2] = cross(tmp[1], tmp[2]);

		float len1 = dot(vec_tmp[0], vec_tmp[0]);
		float len2 = dot(vec_tmp[1], vec_tmp[1]);
		float len3 = dot(vec_tmp[2], vec_tmp[2]);

		if (len1 >= len2 && len1 >= len3)
		{
			evecs[0] = vec_tmp[0] * inv_sqrt(len1);
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			evecs[0] = vec_tmp[1] * inv_sqrt(len2);
		}
		else
		{
			evecs[0] = vec_tmp[2] * inv_sqrt(len3);
		}

		evecs[1] = unit_orthogonal(evecs[0]);
		evecs[2] = cross(evecs[0], evecs[1]);
	}
	else
	{

		tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
		tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

		vec_tmp[0] = cross(tmp[0], tmp[1]);
		vec_tmp[1] = cross(tmp[0], tmp[2]);
		vec_tmp[2] = cross(tmp[1], tmp[2]);

		float len1 = dot(vec_tmp[0], vec_tmp[0]);
		float len2 = dot(vec_tmp[1], vec_tmp[1]);
		float len3 = dot(vec_tmp[2], vec_tmp[2]);

		float mmax[3];

		unsigned int min_el = 2;
		unsigned int max_el = 2;
		if (len1 >= len2 && len1 >= len3)
		{
			mmax[2] = len1;
			evecs[2] = vec_tmp[0] * inv_sqrt(len1);
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			mmax[2] = len2;
			evecs[2] = vec_tmp[1] * inv_sqrt(len2);
		}
		else
		{
			mmax[2] = len3;
			evecs[2] = vec_tmp[2] * inv_sqrt(len3);
		}

		tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
		tmp[0].x -= evals.y; tmp[1].y -= evals.y; tmp[2].z -= evals.y;

		vec_tmp[0] = cross(tmp[0], tmp[1]);
		vec_tmp[1] = cross(tmp[0], tmp[2]);
		vec_tmp[2] = cross(tmp[1], tmp[2]);

		len1 = dot(vec_tmp[0], vec_tmp[0]);
		len2 = dot(vec_tmp[1], vec_tmp[1]);
		len3 = dot(vec_tmp[2], vec_tmp[2]);

		if (len1 >= len2 && len1 >= len3)
		{
			mmax[1] = len1;
			evecs[1] = vec_tmp[0] * inv_sqrt(len1);
			min_el = len1 <= mmax[min_el] ? 1 : min_el;
			max_el = len1  > mmax[max_el] ? 1 : max_el;
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			mmax[1] = len2;
			evecs[1] = vec_tmp[1] * inv_sqrt(len2);
			min_el = len2 <= mmax[min_el] ? 1 : min_el;
			max_el = len2  > mmax[max_el] ? 1 : max_el;
		}
		else
		{
			mmax[1] = len3;
			evecs[1] = vec_tmp[2] * inv_sqrt(len3);
			min_el = len3 <= mmax[min_el] ? 1 : min_el;
			max_el = len3 >  mmax[max_el] ? 1 : max_el;
		}

		tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
		tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

		vec_tmp[0] = cross(tmp[0], tmp[1]);
		vec_tmp[1] = cross(tmp[0], tmp[2]);
		vec_tmp[2] = cross(tmp[1], tmp[2]);

		len1 = dot(vec_tmp[0], vec_tmp[0]);
		len2 = dot(vec_tmp[1], vec_tmp[1]);
		len3 = dot(vec_tmp[2], vec_tmp[2]);


		if (len1 >= len2 && len1 >= len3)
		{
			mmax[0] = len1;
			evecs[0] = vec_tmp[0] * inv_sqrt(len1);
			min_el = len3 <= mmax[min_el] ? 0 : min_el;
			max_el = len3  > mmax[max_el] ? 0 : max_el;
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			mmax[0] = len2;
			evecs[0] = vec_tmp[1] * inv_sqrt(len2);
			min_el = len3 <= mmax[min_el] ? 0 : min_el;
			max_el = len3  > mmax[max_el] ? 0 : max_el;
		}
		else
		{
			mmax[0] = len3;
			evecs[0] = vec_tmp[2] * inv_sqrt(len3);
			min_el = len3 <= mmax[min_el] ? 0 : min_el;
			max_el = len3  > mmax[max_el] ? 0 : max_el;
		}

		unsigned mid_el = 3 - min_el - max_el;
		evecs[min_el] = normalized(cross(evecs[(min_el + 1) % 3], evecs[(min_el + 2) % 3]));
		evecs[mid_el] = normalized(cross(evecs[(mid_el + 1) % 3], evecs[(mid_el + 2) % 3]));
	}
	// Rescale back to the original size.
	evals = evals * scale;
}


__host__ __device__ __forceinline__ void surfelwarp::eigen33::compute(float3 & eigen_vec)
{
	Mat33 tmp, vec_tmp, evecs;
	float3 evals;
	compute(tmp, vec_tmp, evecs, evals);
	eigen_vec = evecs[0];
}

__host__ __device__ __forceinline__ void surfelwarp::eigen33::compute(float3 & eigen_vec, float& eigen_value)
{
	Mat33 tmp, vec_tmp, evecs;
	float3 evals;
	compute(tmp, vec_tmp, evecs, evals);
	eigen_vec = evecs[0];
	eigen_value = evals.x;
}