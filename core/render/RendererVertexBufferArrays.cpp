//
// Created by wei on 3/18/18.
//

#include "core/render/Renderer.h"
#include "core/render/GLSurfelGeometryVBO.h"
#include "core/render/GLSurfelGeometryVAO.h"
#include "common/logging.h"


/* The vertex buffer objects mapping and selection and
 */
void surfelwarp::Renderer::initVertexBufferObjects() {
	initializeGLSurfelGeometry(m_surfel_geometry_vbos[0]);
	initializeGLSurfelGeometry(m_surfel_geometry_vbos[1]);
}

void surfelwarp::Renderer::freeVertexBufferObjects() {
	releaseGLSurfelGeometry(m_surfel_geometry_vbos[0]);
	releaseGLSurfelGeometry(m_surfel_geometry_vbos[1]);
}

void surfelwarp::Renderer::MapSurfelGeometryToCuda(int idx, surfelwarp::SurfelGeometry &geometry, cudaStream_t stream) {
	idx = idx % 2;
	m_surfel_geometry_vbos[idx].mapToCuda(geometry, stream);
}

void surfelwarp::Renderer::MapSurfelGeometryToCuda(int idx, cudaStream_t stream) {
	idx = idx % 2;
	m_surfel_geometry_vbos[idx].mapToCuda(stream);
}

void surfelwarp::Renderer::UnmapSurfelGeometryFromCuda(int idx, cudaStream_t stream) {
	idx = idx % 2;
	m_surfel_geometry_vbos[idx].unmapFromCuda(stream);
}


/* The methods for vertex array objects
 */
void surfelwarp::Renderer::initMapRenderVAO() {
	//Each vao match one for vbos
	buildFusionMapVAO(m_surfel_geometry_vbos[0], m_fusion_map_vao[0]);
	buildFusionMapVAO(m_surfel_geometry_vbos[1], m_fusion_map_vao[1]);
	
	buildSolverMapVAO(m_surfel_geometry_vbos[0], m_solver_map_vao[0]);
	buildSolverMapVAO(m_surfel_geometry_vbos[1], m_solver_map_vao[1]);
	
	buildReferenceGeometryVAO(m_surfel_geometry_vbos[0], m_reference_geometry_vao[0]);
	buildReferenceGeometryVAO(m_surfel_geometry_vbos[1], m_reference_geometry_vao[1]);
	
	buildLiveGeometryVAO(m_surfel_geometry_vbos[0], m_live_geometry_vao[0]);
	buildLiveGeometryVAO(m_surfel_geometry_vbos[1], m_live_geometry_vao[1]);
}
