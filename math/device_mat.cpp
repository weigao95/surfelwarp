//
// Created by wei on 2/8/18.
//

#include "common/common_types.h"
#include "math/device_mat.h"
#include "math/eigen_device_tranfer.h"
#include "math/vector_ops.hpp"

surfelwarp::mat33::mat33(const Eigen::Matrix3f &rhs) {
	cols[0] = make_float3(rhs(0, 0), rhs(1, 0), rhs(2, 0));
	cols[1] = make_float3(rhs(0, 1), rhs(1, 1), rhs(2, 1));
	cols[2] = make_float3(rhs(0, 2), rhs(1, 2), rhs(2, 2));
}

surfelwarp::mat33& surfelwarp::mat33::operator=(const Eigen::Matrix3f &rhs) {
	cols[0] = make_float3(rhs(0, 0), rhs(1, 0), rhs(2, 0));
	cols[1] = make_float3(rhs(0, 1), rhs(1, 1), rhs(2, 1));
	cols[2] = make_float3(rhs(0, 2), rhs(1, 2), rhs(2, 2));
	return *this;
}

surfelwarp::mat34::mat34(const surfelwarp::Isometry3f &se3) : rot(se3.linear().matrix()) {
	Eigen::Vector3f translation = se3.translation();
	trans = fromEigen(translation);
}

surfelwarp::mat34::mat34(const surfelwarp::Matrix4f &rhs) : rot(rhs.block<3, 3>(0, 0)) {
	Eigen::Vector3f eigen_trans = rhs.block<3, 1>(0, 3);
	trans = fromEigen(eigen_trans);
}

surfelwarp::mat34::mat34(const float3 &twist_rot, const float3 &twist_trans) {
	if (fabsf_sum(twist_rot) < 1e-4f) {
		rot.set_identity();
	}
	else {
		float angle = ::surfelwarp::norm(twist_rot);
		float3 axis = (1.0f / angle) * twist_rot;
		
		float c = cosf(angle);
		float s = sinf(angle);
		float t = 1.0f - c;
		
		rot.m00() = t*axis.x*axis.x + c;
		rot.m01() = t*axis.x*axis.y - axis.z*s;
		rot.m02() = t*axis.x*axis.z + axis.y*s;
		
		rot.m10() = t*axis.x*axis.y + axis.z*s;
		rot.m11() = t*axis.y*axis.y + c;
		rot.m12() = t*axis.y*axis.z - axis.x*s;
		
		rot.m20() = t*axis.x*axis.z - axis.y*s;
		rot.m21() = t*axis.y*axis.z + axis.x*s;
		rot.m22() = t*axis.z*axis.z + c;
	}
	
	//The translation part
	trans = twist_trans;
}
