//
// Created by wei on 3/15/18.
//

#include "core/Camera.h"
#include "math/eigen_device_tranfer.h"

surfelwarp::Camera::Camera() {
	m_world2camera = mat34::identity();
	m_camera2world = mat34::identity();
	m_init_camera2world.setIdentity();
	m_init_world2camera.setIdentity();
}

surfelwarp::Camera::Camera(const Eigen::Isometry3f &init_camera2world) {
	m_init_camera2world = init_camera2world.matrix();
	m_init_world2camera = init_camera2world.inverse().matrix();
	m_camera2world = mat34(init_camera2world);
	m_world2camera = mat34(init_camera2world.inverse());
}

void surfelwarp::Camera::SetWorld2Camera(const surfelwarp::mat34 &world2camera) {
	m_world2camera = world2camera;
	m_camera2world = world2camera.inverse();
}

const surfelwarp::mat34 &surfelwarp::Camera::GetWorld2Camera() const {
	return m_world2camera;
}

const surfelwarp::mat34 &surfelwarp::Camera::GetCamera2World() const {
	return m_camera2world;
}

Eigen::Matrix4f surfelwarp::Camera::GetWorld2CameraEigen() const {
	return toEigen(m_world2camera);
}

Eigen::Matrix4f surfelwarp::Camera::GetCamera2WorldEigen() const {
	return toEigen(m_camera2world);
}

const Eigen::Matrix4f &surfelwarp::Camera::GetInitWorld2CameraEigen() const {
	return m_init_world2camera;
}

const Eigen::Matrix4f &surfelwarp::Camera::GetInitCamera2WorldEigen() const {
	return m_init_camera2world;
}


