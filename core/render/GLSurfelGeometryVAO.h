//
// Created by wei on 3/19/18.
//

#pragma once

#include "core/render/GLSurfelGeometryVBO.h"

namespace surfelwarp {
	
	//Init the vao for fusion map
	void buildFusionMapVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& fusion_map_vao);
	
	//Init the vao for warp solver
	void buildSolverMapVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& solver_map_vao);

	//Init the vao for reference geometry
	void buildReferenceGeometryVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& reference_geometry_vao);

	//Init the vao for live geometry
	void buildLiveGeometryVAO(const GLSurfelGeometryVBO& geometryVBO, GLuint& live_geometry_vbo);
}
