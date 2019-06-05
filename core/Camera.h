//
// Created by wei on 3/15/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "math/device_mat.h"

namespace surfelwarp {
	
	class Camera {
	private:
		//The extrinsic parameters of current camera
		mat34 m_world2camera;
		mat34 m_camera2world;
		
		//The initial extrinsic parameters
		Eigen::Matrix4f m_init_world2camera;
		Eigen::Matrix4f m_init_camera2world;
	public:
		Camera();
		Camera(const Eigen::Isometry3f& init_camera2world);
		~Camera() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(Camera);
		
		//The only allowed modified interface, will touch both world2camera and camera2world
		void SetWorld2Camera(const mat34& world2camera);
		
		//The query interface
		const mat34& GetWorld2Camera() const;
		const mat34& GetCamera2World() const;
		
		//These interface for rendering
		Eigen::Matrix4f GetWorld2CameraEigen() const;
		Eigen::Matrix4f GetCamera2WorldEigen() const;
		const Eigen::Matrix4f& GetInitWorld2CameraEigen() const;
		const Eigen::Matrix4f& GetInitCamera2WorldEigen() const;
	};
	
}