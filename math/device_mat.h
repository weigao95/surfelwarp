//
// Created by wei on 2/7/18.
//


#pragma once
#include <vector_functions.h>
#include "common/common_types.h"
#include "math/vector_ops.hpp"

namespace surfelwarp {

    //Simple matrix for access on device (and host)
    
    struct mat33 {
        __host__ __device__ mat33() {}
        __host__ __device__ mat33(const float3 &_a0, const float3 &_a1, const float3 &_a2) { cols[0] = _a0; cols[1] = _a1; cols[2] = _a2; }
        __host__ __device__ mat33(const float *_data)
        {
            /*_data MUST have at least 9 float elements, ctor does not check range*/
            cols[0] = make_float3(_data[0], _data[1], _data[2]);
            cols[1] = make_float3(_data[3], _data[4], _data[5]);
            cols[2] = make_float3(_data[6], _data[7], _data[8]);
        }
	    __host__ mat33(const Eigen::Matrix3f& matrix3f);
	    __host__ mat33& operator=(const Eigen::Matrix3f& matrix3f);
        
        __host__ __device__ const float& m00() const { return cols[0].x; }
        __host__ __device__ const float& m10() const { return cols[0].y; }
        __host__ __device__ const float& m20() const { return cols[0].z; }
        __host__ __device__ const float& m01() const { return cols[1].x; }
        __host__ __device__ const float& m11() const { return cols[1].y; }
        __host__ __device__ const float& m21() const { return cols[1].z; }
        __host__ __device__ const float& m02() const { return cols[2].x; }
        __host__ __device__ const float& m12() const { return cols[2].y; }
        __host__ __device__ const float& m22() const { return cols[2].z; }
        
        __host__ __device__ float& m00() { return cols[0].x; }
        __host__ __device__ float& m10() { return cols[0].y; }
        __host__ __device__ float& m20() { return cols[0].z; }
        __host__ __device__ float& m01() { return cols[1].x; }
        __host__ __device__ float& m11() { return cols[1].y; }
        __host__ __device__ float& m21() { return cols[1].z; }
        __host__ __device__ float& m02() { return cols[2].x; }
        __host__ __device__ float& m12() { return cols[2].y; }
        __host__ __device__ float& m22() { return cols[2].z; }
        
        __host__ __device__ mat33 transpose() const
        {
            float3 row0 = make_float3(cols[0].x, cols[1].x, cols[2].x);
            float3 row1 = make_float3(cols[0].y, cols[1].y, cols[2].y);
            float3 row2 = make_float3(cols[0].z, cols[1].z, cols[2].z);
            return mat33(row0, row1, row2);
        }
        
        __host__ __device__ mat33 operator* (const mat33 &_mat) const
        {
            mat33 mat;
            mat.m00() = m00()*_mat.m00() + m01()*_mat.m10() + m02()*_mat.m20();
            mat.m01() = m00()*_mat.m01() + m01()*_mat.m11() + m02()*_mat.m21();
            mat.m02() = m00()*_mat.m02() + m01()*_mat.m12() + m02()*_mat.m22();
            mat.m10() = m10()*_mat.m00() + m11()*_mat.m10() + m12()*_mat.m20();
            mat.m11() = m10()*_mat.m01() + m11()*_mat.m11() + m12()*_mat.m21();
            mat.m12() = m10()*_mat.m02() + m11()*_mat.m12() + m12()*_mat.m22();
            mat.m20() = m20()*_mat.m00() + m21()*_mat.m10() + m22()*_mat.m20();
            mat.m21() = m20()*_mat.m01() + m21()*_mat.m11() + m22()*_mat.m21();
            mat.m22() = m20()*_mat.m02() + m21()*_mat.m12() + m22()*_mat.m22();
            return mat;
        }
        
        __host__ __device__ mat33 operator+ (const mat33 &_mat) const
        {
            mat33 mat_sum;
            mat_sum.m00() = m00() + _mat.m00();
            mat_sum.m01() = m01() + _mat.m01();
            mat_sum.m02() = m02() + _mat.m02();
            
            mat_sum.m10() = m10() + _mat.m10();
            mat_sum.m11() = m11() + _mat.m11();
            mat_sum.m12() = m12() + _mat.m12();
            
            mat_sum.m20() = m20() + _mat.m20();
            mat_sum.m21() = m21() + _mat.m21();
            mat_sum.m22() = m22() + _mat.m22();
            
            return mat_sum;
        }
        
        __host__ __device__ mat33 operator- (const mat33 &_mat) const
        {
            mat33 mat_diff;
            mat_diff.m00() = m00() - _mat.m00();
            mat_diff.m01() = m01() - _mat.m01();
            mat_diff.m02() = m02() - _mat.m02();
            
            mat_diff.m10() = m10() - _mat.m10();
            mat_diff.m11() = m11() - _mat.m11();
            mat_diff.m12() = m12() - _mat.m12();
            
            mat_diff.m20() = m20() - _mat.m20();
            mat_diff.m21() = m21() - _mat.m21();
            mat_diff.m22() = m22() - _mat.m22();
            
            return mat_diff;
        }
        
        __host__ __device__ mat33 operator-() const
        {
            mat33 mat_neg;
            mat_neg.m00() = -m00();
            mat_neg.m01() = -m01();
            mat_neg.m02() = -m02();
            
            mat_neg.m10() = -m10();
            mat_neg.m11() = -m11();
            mat_neg.m12() = -m12();
            
            mat_neg.m20() = -m20();
            mat_neg.m21() = -m21();
            mat_neg.m22() = -m22();
            
            return mat_neg;
        }
        
        __host__ __device__ mat33& operator*= (const mat33 &_mat)
        {
            *this = *this * _mat;
            return *this;
        }
        
        __host__ __device__ float3 operator* (const float3 &_vec) const
        {
            const float x = m00()*_vec.x + m01()*_vec.y + m02()*_vec.z;
            const float y = m10()*_vec.x + m11()*_vec.y + m12()*_vec.z;
            const float z = m20()*_vec.x + m21()*_vec.y + m22()*_vec.z;
            return make_float3(x, y, z);
        }

		//Just ignore the vec.w elements
		__host__ __device__ float3 operator* (const float4 &_vec) const
        {
            const float x = m00()*_vec.x + m01()*_vec.y + m02()*_vec.z;
            const float y = m10()*_vec.x + m11()*_vec.y + m12()*_vec.z;
            const float z = m20()*_vec.x + m21()*_vec.y + m22()*_vec.z;
            return make_float3(x, y, z);
        }

		//Conceptually, transpose this matrix and perform a dot product
		__host__ __device__ float3 transpose_dot(const float3& _vec) const
        {
	        const float x = m00()*_vec.x + m10()*_vec.y + m20()*_vec.z;
            const float y = m01()*_vec.x + m11()*_vec.y + m21()*_vec.z;
            const float z = m02()*_vec.x + m12()*_vec.y + m22()*_vec.z;
            return make_float3(x, y, z);
        }

		__host__ __device__ float3 transpose_dot(const float4& _vec) const
        {
	        const float x = m00()*_vec.x + m10()*_vec.y + m20()*_vec.z;
            const float y = m01()*_vec.x + m11()*_vec.y + m21()*_vec.z;
            const float z = m02()*_vec.x + m12()*_vec.y + m22()*_vec.z;
            return make_float3(x, y, z);
        }
        
        __host__ __device__ void set_identity()
        {
            cols[0] = make_float3(1, 0, 0);
            cols[1] = make_float3(0, 1, 0);
            cols[2] = make_float3(0, 0, 1);
        }
        
        __host__ __device__ static mat33 identity()
        {
            mat33 idmat;
            idmat.set_identity();
            return idmat;
        }
        
        float3 cols[3]; /*colume major*/
    };

    //Simple SE(3) matrix representation
    struct mat34 {
	    __host__ __device__ mat34() {}
	    __host__ __device__ mat34(const mat33 &_rot, const float3 &_trans) : rot(_rot), trans(_trans) {}
	    __host__ __device__ mat34(const float3& twist_rot, const float3& twist_trans);
	    __host__ __device__ static mat34 identity()
	    {
		    return mat34(mat33::identity(), make_float3(0, 0, 0));
	    }
        __host__ mat34(const Isometry3f& se3);
	    __host__ mat34(const Matrix4f& matrix4f);
	
	    __host__ __device__ mat34 operator* (const mat34 &_right_se3) const
	    {
		    mat34 se3;
		    se3.rot = rot*_right_se3.rot;
		    se3.trans = (rot*_right_se3.trans) + trans;
		    return se3;
	    }
	
	    __host__ __device__ mat34& operator*= (const mat34 &_right_se3)
	    {
		    *this = *this * _right_se3;
		    return *this;
	    }
	    
	    __host__ __device__ __forceinline__ void set_identity() {
		    rot.set_identity();
		    trans.x = trans.y = trans.z = 0.0f;
	    }

		__host__ __device__ __forceinline__ mat34 inverse() const {
		    mat34 inversed;
			inversed.rot = rot.transpose();
			inversed.trans = - (inversed.rot * trans);
			return inversed;
	    }

		__host__ __device__ __forceinline__ float3 apply_inversed_se3(const float3& vec) const {
		    return rot.transpose_dot(vec - trans);
	    }

		//Just ignore the last elements
		__host__ __device__ __forceinline__ float3 apply_inversed_se3(const float4& vec) const {
		    return rot.transpose_dot(make_float3(
				vec.x - trans.x, 
				vec.y - trans.y, 
				vec.z - trans.z
			));
	    }
	
	    mat33 rot;
	    float3 trans;
    };
	
	
}