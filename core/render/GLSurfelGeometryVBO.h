//
// Created by wei on 3/18/18.
//

#pragma once

#include "core/render/glad/glad.h"
#include "core/SurfelGeometry.h"

namespace surfelwarp {
	
	/**
	 * \brief A struct to maintain all the
	 *        vertex buffer objects for a 
	 *        instance of surfel geometry, 
	 *        alongside with reource for
	 *        access on cuda.
	 *        This struct can only be used
	 *        inside the render class.
	 */
	struct GLSurfelGeometryVBO {
		//The vertex buffer objects correspond
		//to the member of SurfelGeometry
		GLuint reference_vertex_confid;
		GLuint reference_normal_radius;
		GLuint live_vertex_confid;
		GLuint live_normal_radius;
		GLuint color_time;

		//The cuda resource associated
		//with the surfel geomety vbos
		cudaGraphicsResource_t cuda_vbo_resources[5];
		
		//Methods can only be accessed by renderer
		void initialize();
		void release();
		
		void mapToCuda(SurfelGeometry& geometry, cudaStream_t stream = 0);
		void mapToCuda(cudaStream_t stream = 0);
		
		//block all cuda calls in the given threads
		//for later OpenGL drawing pipelines
		void unmapFromCuda(cudaStream_t stream = 0);
	};
	
	//Use procedual for more clear ordering
	void initializeGLSurfelGeometry(GLSurfelGeometryVBO& surfel_vbo);
	void releaseGLSurfelGeometry(GLSurfelGeometryVBO& surfel_vbo);
}