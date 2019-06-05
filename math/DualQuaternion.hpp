#pragma once
#include "math/Quaternion.hpp"
#include "math/vector_ops.hpp"

namespace surfelwarp {

	//The DualNumber associated with DualQuaternion
	struct DualNumber {
		__host__ __device__ DualNumber() : q0(0), q1(0) {}
		__host__ __device__ DualNumber(float _q0, float _q1) : q0(_q0), q1(_q1) {}
		
		__host__ __device__ DualNumber operator+(const DualNumber &_dn) const
		{
			return{ q0 + _dn.q0, q1 + _dn.q1 };
		}
		
		__host__ __device__ DualNumber& operator+=(const DualNumber &_dn)
		{
			*this = *this + _dn;
			return *this;
		}
		
		__host__ __device__ DualNumber operator*(const DualNumber &_dn) const
		{
			return{ q0*_dn.q0, q0*_dn.q1 + q1*_dn.q0 };
		}
		
		__host__ __device__ DualNumber& operator*=(const DualNumber &_dn)
		{
			*this = *this * _dn;
			return *this;
		}
		
		__host__ __device__ DualNumber reciprocal() const
		{
			return{ 1.0f / q0, -q1 / (q0*q0) };
		}
		
		__host__ __device__ DualNumber sqrt() const
		{
			return{ sqrtf(q0), q1 / (2 * sqrtf(q0)) };
		}
		
		float q0, q1;
	};
	
	
	// Forward declaration
	struct DualQuaternion;
	__host__ __device__ DualQuaternion operator*(const DualNumber &_dn, const DualQuaternion &_dq);
	
	
	struct DualQuaternion {
		
		__host__ __device__ DualQuaternion() {}
		__host__ __device__ DualQuaternion(const Quaternion &_q0, const Quaternion &_q1) : q0(_q0), q1(_q1) {}
		__host__ __device__ DualQuaternion(const mat34 &T)
		{
			mat33 r = T.rot;
			float3 t = T.trans;
			DualQuaternion rot_part(Quaternion(r), Quaternion(0, 0, 0, 0));
			DualQuaternion vec_part(Quaternion(1, 0, 0, 0), Quaternion(0, 0.5f*t.x, 0.5f*t.y, 0.5f*t.z));
			*this = vec_part * rot_part;
		}
		
		__host__ __device__ DualQuaternion operator+(const DualQuaternion &_dq) const
		{
			Quaternion quat0(q0 + _dq.q0);
			Quaternion quat1(q1 + _dq.q1);
			return{ quat0, quat1 };
		}
		
		__host__ __device__ DualQuaternion operator*(const DualQuaternion &_dq) const
		{
			Quaternion quat0(q0*_dq.q0);
			Quaternion quat1(q1*_dq.q0 + q0*_dq.q1);
			return{ quat0, quat1 };
		}
		
		__host__ __device__ DualQuaternion operator*(const float &_w) const
		{
			return{ _w * q0, _w * q1 };
		}
		
		__host__ __device__ float3 operator*(const float3 &_p) const
		{
			float3 vec0 = q0.vec();
			float3 vec1 = q1.vec();
			return _p + 2 * (cross(vec0, cross(vec0, _p) + q0.w() * _p) + vec1 * q0.w() - vec0 * q1.w() + cross(vec0, vec1));
		}
		
		__host__ __device__ float3 rotate(const float3 &_p) const
		{
			float3 vec0 = q0.vec();
			return _p + 2 * cross(vec0, cross(vec0, _p) + q0.w() * _p);
		}
		
		__host__ __device__ DualQuaternion& operator+=(const DualQuaternion &_dq)
		{
			*this = *this + _dq;
			return *this;
		}
		
		__host__ __device__ DualQuaternion& operator*=(const DualQuaternion &_dq)
		{
			*this = *this * _dq;
			return *this;
		}
		
		__host__ __device__ DualQuaternion operator*(const DualNumber &_dn) const
		{
			return _dn * *this;
		}
		
		__host__ __device__ DualQuaternion& operator*=(const DualNumber &_dn)
		{
			*this = *this * _dn;
			return *this;
		}
		
		__host__ __device__ operator DualNumber() const
		{
			return DualNumber(q0.w(), q1.w());
		}
		
		__host__ __device__ DualQuaternion conjugate() const
		{
			return{ q0.conjugate(), q1.conjugate() };
		}
		
		__host__ __device__ DualNumber squared_norm() const
		{
			return (*this) * (this->conjugate());
		}
		
		__host__ __device__ DualNumber norm() const
		{
			float a0 = q0.norm();
			float a1 = q0.dot(q1) / q0.norm();
			return{ a0, a1 };
		}
		
		__host__ __device__ DualQuaternion inverse() const
		{
			return this->conjugate() * this->squared_norm().reciprocal();
		}
		
		__host__ __device__ void normalize()
		{
			const float inv_norm = q0.norm_inversed();
			q0 = inv_norm * q0;
			q1 = inv_norm * q1;
			q1 = q1 - dot(q0, q1) * q0;
		}
		__host__ __device__ void normalize_indirect()
		{
			*this = *this * this->norm().reciprocal();
		}
		
		__host__ __device__ DualQuaternion normalized() const {
			DualQuaternion dq = *this;
			dq.normalize();
			return dq;
		}
		
		__host__ __device__ operator mat34() const
		{
			mat33 r;
			float3 t;
			DualQuaternion quat_normalized = this->normalized();
			r = quat_normalized.q0.matrix();
			Quaternion vec_part = 2.0f*quat_normalized.q1*quat_normalized.q0.conjugate();
			t = vec_part.vec();
			
			return mat34(r, t);
		}
		
		//This might mutate the value
		__host__ __device__ mat34 se3_matrix() {
			this->normalize();
			const mat33 rotate = this->q0.rotation_matrix(false);
			const Quaternion trans_part = 2.0f * q1 * q0.conjugate();
			const float3 translate = make_float3(trans_part.x(), trans_part.y(), trans_part.z());
			return mat34(rotate, translate);
		}
		
		//Use this method when the quaternion is used for average
		//After calling this, dont use normalized
		__host__ __device__ void set_zero() {
			q0.x() = q0.y() = q0.z() = q0.w() = 0.f;
			q1.x() = q1.y() = q1.z() = q1.w() = 0.f;
		}
		
		__host__ __device__ void set_identity() {
			q0.w() = 1.0f;
			q0.x() = q0.y() = q0.z() = 0.f;
			q1.x() = q1.y() = q1.z() = q1.w() = 0.f;
		}
		
		Quaternion q0, q1;
	};
	
	
	__host__ __device__ __forceinline__ DualQuaternion operator*(const DualNumber &_dn, const DualQuaternion &_dq)
	{
		const Quaternion quat0 = _dn.q0*_dq.q0;
		const Quaternion quat1 = _dn.q0*_dq.q1 + _dn.q1*_dq.q0;
		return{ quat0, quat1 };
	}

	__host__ __device__ __forceinline__ DualQuaternion averageDualQuaternion(
		const DualQuaternion* warp_field,
		const ushort4& knn,
		const float4& weight
	) {
		DualQuaternion dq_average; dq_average.set_zero();
		dq_average += DualNumber(weight.x, 0) * warp_field[knn.x];
		dq_average += DualNumber(weight.y, 0) * warp_field[knn.y];
		dq_average += DualNumber(weight.z, 0) * warp_field[knn.z];
		dq_average += DualNumber(weight.w, 0) * warp_field[knn.w];
		return dq_average;
	}

	__host__ __device__ __forceinline__ DualQuaternion averageDualQuaternion(
		const DualQuaternion* warp_field,
		const int4& knn,
		const float4& weight
	) {
		DualQuaternion dq_average; dq_average.set_zero();
		dq_average += DualNumber(weight.x, 0) * warp_field[knn.x];
		dq_average += DualNumber(weight.y, 0) * warp_field[knn.y];
		dq_average += DualNumber(weight.z, 0) * warp_field[knn.z];
		dq_average += DualNumber(weight.w, 0) * warp_field[knn.w];
		return dq_average;
	}
	
	//Perform left product with the given twist
	__host__ __device__ __forceinline__ void apply_twist(
		const float3& twist_rot,
		const float3& twist_trans,
		DualQuaternion& dq
	) {
		mat34 SE3;
		if (fabsf_sum(twist_rot) < 1e-4f) {
			SE3.rot.set_identity();
		}
		else {
			float angle = ::surfelwarp::norm(twist_rot);
			float3 axis = (1.0f / angle) * twist_rot;
			
			float c = cosf(angle);
			float s = sinf(angle);
			float t = 1.0f - c;
			
			SE3.rot.m00() = t*axis.x*axis.x + c;
			SE3.rot.m01() = t*axis.x*axis.y - axis.z*s;
			SE3.rot.m02() = t*axis.x*axis.z + axis.y*s;
			
			SE3.rot.m10() = t*axis.x*axis.y + axis.z*s;
			SE3.rot.m11() = t*axis.y*axis.y + c;
			SE3.rot.m12() = t*axis.y*axis.z - axis.x*s;
			
			SE3.rot.m20() = t*axis.x*axis.z - axis.y*s;
			SE3.rot.m21() = t*axis.y*axis.z + axis.x*s;
			SE3.rot.m22() = t*axis.z*axis.z + c;
		}
		
		SE3.trans = twist_trans;
		dq = SE3 * dq;
	}
}
