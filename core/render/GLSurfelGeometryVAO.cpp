//
// Created by wei on 3/19/18.
//

#include "core/render/GLSurfelGeometryVAO.h"

void surfelwarp::buildFusionMapVAO(const surfelwarp::GLSurfelGeometryVBO &geometryVBO, GLuint &fusion_map_vao) {
	//Create and bind vao
	glGenVertexArrays(1, &fusion_map_vao);
	glBindVertexArray(fusion_map_vao);
	
	//First array is warpped vertex array
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.live_vertex_confid);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);
	
	//Next is the warpped normal array
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.live_normal_radius);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);
	
	//And the color time array
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.color_time);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);
	
	//Cleanup code
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void surfelwarp::buildSolverMapVAO(const surfelwarp::GLSurfelGeometryVBO &geometryVBO, GLuint &solver_map_vao) {
	//Create and bind vao
	glGenVertexArrays(1, &solver_map_vao);
	glBindVertexArray(solver_map_vao);
	
	//Need to bind all the attributes
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.reference_vertex_confid);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);
	
	//The reference normal radius buffer
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.reference_normal_radius);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);
	
	//The warpped vertex confidence
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.live_vertex_confid);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);
	
	//The warpped normal array
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.live_normal_radius);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(3);
	
	//And the color time array
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.color_time);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(4);
	
	//Cleanup code
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}


void surfelwarp::buildReferenceGeometryVAO(const GLSurfelGeometryVBO & geometryVBO, GLuint & reference_geometry_vao) {
	//Create and bind vao
	glGenVertexArrays(1, &reference_geometry_vao);
	glBindVertexArray(reference_geometry_vao);

	//Need to bind all the attributes
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.reference_vertex_confid);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(0);
	
	//The reference normal radius buffer
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.reference_normal_radius);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(1);

	//And the color time array
	glBindBuffer(GL_ARRAY_BUFFER, geometryVBO.color_time);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glEnableVertexAttribArray(2);

	//Cleanup code
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void surfelwarp::buildLiveGeometryVAO(const GLSurfelGeometryVBO & geometryVBO, GLuint & live_geometry_vbo) {
	//The fusion map and vao use the same vertex buffers
	buildFusionMapVAO(geometryVBO, live_geometry_vbo);
}
