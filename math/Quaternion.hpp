#pragma once

#include "common/common_types.h"
#include "math/device_mat.h"
#include "math/vector_ops.hpp"

namespace surfelwarp {

    /*
     * Quaternion struct for cuda access
     * */
    struct Quaternion
    {
	    __host__ __device__ Quaternion() {}
	    __host__ __device__ Quaternion(float _w, float _x, float _y, float _z) : q0(make_float4(_x, _y, _z, _w)) {}
	    __host__ __device__ Quaternion(const float4 &_q) : q0(_q) {}
	    __host__ __device__ Quaternion(const mat33 &_rot)
	    {
		    float tr = _rot.m00() + _rot.m11() + _rot.m22();
		    if (tr > 0) {
			    float s = sqrtf(tr + 1.0f) * 2;
			    q0.w = s*0.25f;
			    q0.x = (_rot.m21() - _rot.m12()) / s;
			    q0.y = (_rot.m02() - _rot.m20()) / s;
			    q0.z = (_rot.m10() - _rot.m01()) / s;
		    }
		    else if ((_rot.m00() > _rot.m11()) && (_rot.m00() > _rot.m22())) {
			    float s = sqrtf(1.0f + _rot.m00() - _rot.m11() - _rot.m22()) * 2;
			    q0.w = (_rot.m21() - _rot.m12()) / s;
			    q0.x = 0.25f*s;
			    q0.y = (_rot.m01() + _rot.m10()) / s;
			    q0.z = (_rot.m02() + _rot.m20()) / s;
		    }
		    else if (_rot.m11() > _rot.m22()) {
			    float s = sqrtf(1.0f + _rot.m11() - _rot.m00() - _rot.m22()) * 2;
			    q0.w = (_rot.m02() - _rot.m20()) / s;
			    q0.x = (_rot.m01() + _rot.m10()) / s;
			    q0.y = 0.25f*s;
			    q0.z = (_rot.m12() + _rot.m21()) / s;
		    }
		    else {
			    float s = sqrtf(1.0f + _rot.m22() - _rot.m00() - _rot.m11()) * 2;
			    q0.w = (_rot.m10() - _rot.m01()) / s;
			    q0.x = (_rot.m02() + _rot.m20()) / s;
			    q0.y = (_rot.m12() + _rot.m21()) / s;
			    q0.z = 0.25f*s;
		    }
	    }
	
	    __host__ __device__ float& x() { return q0.x; }
	    __host__ __device__ float& y() { return q0.y; }
	    __host__ __device__ float& z() { return q0.z; }
	    __host__ __device__ float& w() { return q0.w; }
	
	    __host__ __device__ const float& x() const { return q0.x; }
	    __host__ __device__ const float& y() const { return q0.y; }
	    __host__ __device__ const float& z() const { return q0.z; }
	    __host__ __device__ const float& w() const { return q0.w; }
	
	    __host__ __device__ Quaternion conjugate() const { return Quaternion(q0.w, -q0.x, -q0.y, -q0.z); }
	    __host__ __device__ float square_norm() const { return q0.w*q0.w + q0.x*q0.x + q0.y*q0.y + q0.z*q0.z; }
	    __host__ __device__ float norm() const { return sqrtf(square_norm()); }
	    __host__ __device__ float norm_inversed() const { return ::surfelwarp::norm_inversed(q0); }
	    __host__ __device__ float dot(const Quaternion &_quat) const { return q0.w*_quat.w() + q0.x*_quat.x() + q0.y*_quat.y() + q0.z*_quat.z(); }
	    __host__ __device__ void normalize() { ::surfelwarp::normalize(q0); }
	    __host__ __device__ Quaternion normalized() const { Quaternion q(*this); q.normalize(); return q; }
	
	    __host__ __device__ mat33 matrix() const
	    {
		    /*normalize quaternion before converting to so3 matrix*/
		    Quaternion q(*this);
		    q.normalize();
		
		    mat33 rot;
		    rot.m00() = 1 - 2 * q.y()*q.y() - 2 * q.z()*q.z();
		    rot.m01() = 2 * q.x()*q.y() - 2 * q.z()*q.w();
		    rot.m02() = 2 * q.x()*q.z() + 2 * q.y()*q.w();
		    rot.m10() = 2 * q.x()*q.y() + 2 * q.z()*q.w();
		    rot.m11() = 1 - 2 * q.x()*q.x() - 2 * q.z()*q.z();
		    rot.m12() = 2 * q.y()*q.z() - 2 * q.x()*q.w();
		    rot.m20() = 2 * q.x()*q.z() - 2 * q.y()*q.w();
		    rot.m21() = 2 * q.y()*q.z() + 2 * q.x()*q.w();
		    rot.m22() = 1 - 2 * q.x()*q.x() - 2 * q.y()*q.y();
		    return rot;
	    }
	    
	    __host__ __device__ mat33 rotation_matrix(bool normalize) {
		    if(normalize) this->normalize();
		    mat33 rot;
		    rot.m00() = 1 - 2 * y()*y() - 2 * z()*z();
		    rot.m01() = 2 * x()*y() - 2 * z()*w();
		    rot.m02() = 2 * x()*z() + 2 * y()*w();
		    rot.m10() = 2 * x()*y() + 2 * z()*w();
		    rot.m11() = 1 - 2 * x()*x() - 2 * z()*z();
		    rot.m12() = 2 * y()*z() - 2 * x()*w();
		    rot.m20() = 2 * x()*z() - 2 * y()*w();
		    rot.m21() = 2 * y()*z() + 2 * x()*w();
		    rot.m22() = 1 - 2 * x()*x() - 2 * y()*y();
		    return rot;
	    }
	
	    __host__ __device__ float3 vec() const { return make_float3(q0.x, q0.y, q0.z); }
	
	    float4 q0;
    };


	//Other operators
	__host__ __device__ __forceinline__ Quaternion operator+(const Quaternion &_left, const Quaternion &_right)
	{
		return{ _left.w() + _right.w(), _left.x() + _right.x(), _left.y() + _right.y(), _left.z() + _right.z() };
	}
	
	__host__ __device__ __forceinline__ Quaternion operator-(const Quaternion &_left, const Quaternion &_right) {
		return{ _left.w() - _right.w(), _left.x() - _right.x(), _left.y() - _right.y(), _left.z() - _right.z() };
	}
	
	__host__ __device__ __forceinline__ Quaternion operator*(float _scalar, const Quaternion &_quat)
	{
		return{ _scalar*_quat.w(), _scalar*_quat.x(), _scalar*_quat.y(), _scalar*_quat.z() };
	}
	
	__host__ __device__ __forceinline__ Quaternion operator*(const Quaternion &_quat, float _scalar)
	{
		return _scalar * _quat;
	}
	
	__host__ __device__ __forceinline__ Quaternion operator*(const Quaternion &_q0, const Quaternion &_q1)
	{
		Quaternion q;
		q.w() = _q0.w()*_q1.w() - _q0.x()*_q1.x() - _q0.y()*_q1.y() - _q0.z()*_q1.z();
		q.x() = _q0.w()*_q1.x() + _q0.x()*_q1.w() + _q0.y()*_q1.z() - _q0.z()*_q1.y();
		q.y() = _q0.w()*_q1.y() - _q0.x()*_q1.z() + _q0.y()*_q1.w() + _q0.z()*_q1.x();
		q.z() = _q0.w()*_q1.z() + _q0.x()*_q1.y() - _q0.y()*_q1.x() + _q0.z()*_q1.w();
		
		return q;
	}
	
	__host__ __device__ __forceinline__ float dot(const Quaternion& q0, const Quaternion& q1) {
		return dot(q0.q0, q1.q0);
	}
}