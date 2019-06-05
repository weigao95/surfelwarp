//
// Created by wei on 3/19/18.
//

#pragma once
#include "core/render/glad/glad.h"
#include <cstring>

namespace surfelwarp {
	
	//A small struct to hold the clear values
	struct GLClearValues {
		GLfloat vertex_map_clear[4];
		GLfloat normal_map_clear[4];
		GLfloat color_time_clear[4];
		GLfloat solver_rgba_clear[4];
		GLfloat visualize_rgba_clear[4];
		GLfloat z_buffer_clear;
		GLuint index_map_clear;
		
		//set the values
		inline void initialize() {
			memset(vertex_map_clear, 0, sizeof(GLfloat) * 4);
			memset(normal_map_clear, 0, sizeof(GLfloat) * 4);
			memset(color_time_clear, 0, sizeof(GLfloat) * 4);
			memset(solver_rgba_clear, 0, sizeof(GLfloat) * 4);
			visualize_rgba_clear[0] = 1.0f;
			visualize_rgba_clear[1] = 1.0f;
			visualize_rgba_clear[2] = 1.0f;
			visualize_rgba_clear[3] = 1.0f;
			z_buffer_clear = 1;
			index_map_clear = 0xFFFFFFFF;
		}
	};
}
