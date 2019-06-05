//
// Created by wei on 2/7/18.
//

#include "math/eigen_device_tranfer.h"

Eigen::Matrix3f surfelwarp::toEigen(const mat33 &rhs) {
    Matrix3f lhs;
    lhs(0, 0) = rhs.m00();
    lhs(0, 1) = rhs.m01();
    lhs(0, 2) = rhs.m02();
    lhs(1, 0) = rhs.m10();
    lhs(1, 1) = rhs.m11();
    lhs(1, 2) = rhs.m12();
    lhs(2, 0) = rhs.m20();
    lhs(2, 1) = rhs.m21();
    lhs(2, 2) = rhs.m22();
    return lhs;
}

Eigen::Matrix4f surfelwarp::toEigen(const mat34 &rhs) {
    Matrix4f lhs;
    lhs.setIdentity();
    //The rotational part
    lhs(0, 0) = rhs.rot.m00();
    lhs(0, 1) = rhs.rot.m01();
    lhs(0, 2) = rhs.rot.m02();
    lhs(1, 0) = rhs.rot.m10();
    lhs(1, 1) = rhs.rot.m11();
    lhs(1, 2) = rhs.rot.m12();
    lhs(2, 0) = rhs.rot.m20();
    lhs(2, 1) = rhs.rot.m21();
    lhs(2, 2) = rhs.rot.m22();
    //The translation part
    lhs.block<3, 1>(0, 3) = toEigen(rhs.trans);
    return lhs;
}


Eigen::Vector3f surfelwarp::toEigen(const float3 &rhs) {
    Vector3f lhs;
    lhs(0) = rhs.x;
    lhs(1) = rhs.y;
    lhs(2) = rhs.z;
    return lhs;
}

Eigen::Vector4f surfelwarp::toEigen(const float4 &rhs) {
    Vector4f lhs;
    lhs(0) = rhs.x;
    lhs(1) = rhs.y;
    lhs(2) = rhs.z;
    lhs(3) = rhs.w;
    return lhs;
}

float3 surfelwarp::fromEigen(const Vector3f &rhs) {
    float3 lhs;
    lhs.x = rhs(0);
    lhs.y = rhs(1);
    lhs.z = rhs(2);
    return lhs;
}

float4 surfelwarp::fromEigen(const Vector4f &rhs) {
    float4 lhs;
    lhs.x = rhs(0);
    lhs.y = rhs(1);
    lhs.z = rhs(2);
    lhs.w = rhs(3);
    return lhs;
}


void surfelwarp::fromEigen(
        const Isometry3f &se3,
        Quaternion &rotation,
        float3 &translation
) {
    mat33 rot(se3.linear().matrix());
    rotation = Quaternion(rot);
    Vector3f trans_eigen = se3.translation();
    translation = fromEigen(trans_eigen);
}