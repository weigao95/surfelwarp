//
// Created by wei on 4/4/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "math/DualQuaternion.hpp"

namespace surfelwarp {
	
	class GradientCheck {
		//The geometry data
		std::vector<float4> m_reference_vertex_confid;
		std::vector<float4> m_reference_normal_radius;
		std::vector<ushort4> m_vertex_knn;
		std::vector<float4> m_vertex_knn_weight;
		
		//The warp field data
		std::vector<ushort2> m_node_graph;
		std::vector<float4> m_node_coordinates;

		//Data that do not need to be loaded
		std::vector<DualQuaternion> m_init_node_se3;
		mat34 m_camera2world;
		
	public:
		SURFELWARP_DEFAULT_CONSTRUCT_DESTRUCT(GradientCheck);
		SURFELWARP_NO_COPY_ASSIGN_MOVE(GradientCheck);
		void LoadDataFromFile(const std::string& path);
		void randomInitWarpField(float max_rot = 0.1f, float max_trans = 0.1f);

		//Check the smooth terms
		void checkSmoothTermJacobian();
		
		//Check the point-to-plane term
		void checkPoint2PlaneICPJacobian();
		void checkPoint2PointICPJacobian();
	};
	
}
