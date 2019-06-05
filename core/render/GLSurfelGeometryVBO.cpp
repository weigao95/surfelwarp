//
// Created by wei on 3/18/18.
//

#include "common/common_types.h"
#include "common/Constants.h"
#include "core/render/GLSurfelGeometryVBO.h"

//Register buffer to cuda
#include <cuda_gl_interop.h>

void surfelwarp::GLSurfelGeometryVBO::initialize() {
	initializeGLSurfelGeometry(*this);
}

void surfelwarp::GLSurfelGeometryVBO::release() {
	releaseGLSurfelGeometry(*this);
}

void surfelwarp::GLSurfelGeometryVBO::mapToCuda(
	surfelwarp::SurfelGeometry &geometry,
	cudaStream_t stream
) {
	//First map the resource
	cudaSafeCall(cudaGraphicsMapResources(5, cuda_vbo_resources, stream));
	
	//Get the buffer
	void* dptr;
	size_t buffer_size;
	const size_t num_valid_surfels = geometry.NumValidSurfels();
	
	//reference vertex-confidence
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&dptr, &buffer_size, cuda_vbo_resources[0]));
	geometry.m_reference_vertex_confid = DeviceSliceBufferArray<float4>((float4*)dptr, buffer_size / sizeof(float4), num_valid_surfels);
	
	//reference normal-radius
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&dptr, &buffer_size, cuda_vbo_resources[1]));
	geometry.m_reference_normal_radius = DeviceSliceBufferArray<float4>((float4*)dptr, buffer_size / sizeof(float4), num_valid_surfels);
	
	//live vertex confidence
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&dptr, &buffer_size, cuda_vbo_resources[2]));
	geometry.m_live_vertex_confid = DeviceSliceBufferArray<float4>((float4*)dptr, buffer_size / sizeof(float4), num_valid_surfels);
	
	//live normal-radius
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&dptr, &buffer_size, cuda_vbo_resources[3]));
	geometry.m_live_normal_radius = DeviceSliceBufferArray<float4>((float4*)dptr, buffer_size / sizeof(float4), num_valid_surfels);
	
	//color time
	cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&dptr, &buffer_size, cuda_vbo_resources[4]));
	geometry.m_color_time = DeviceSliceBufferArray<float4>((float4*)dptr, buffer_size / sizeof(float4), num_valid_surfels);
}

void surfelwarp::GLSurfelGeometryVBO::mapToCuda(cudaStream_t stream) {
	cudaSafeCall(cudaGraphicsMapResources(5, cuda_vbo_resources, stream));
}

void surfelwarp::GLSurfelGeometryVBO::unmapFromCuda(cudaStream_t stream) {
	cudaSafeCall(cudaGraphicsUnmapResources(5, cuda_vbo_resources, stream));
}


void surfelwarp::initializeGLSurfelGeometry(surfelwarp::GLSurfelGeometryVBO &vbo) {
	glGenBuffers(1, &(vbo.reference_vertex_confid));
	glGenBuffers(1, &(vbo.reference_normal_radius));
	glGenBuffers(1, &(vbo.live_vertex_confid));
	glGenBuffers(1, &(vbo.live_normal_radius));
	glGenBuffers(1, &(vbo.color_time));
	
	//Allocate buffers
	glBindBuffer(GL_ARRAY_BUFFER, vbo.reference_vertex_confid);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::kMaxNumSurfels, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, vbo.reference_normal_radius);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::kMaxNumSurfels, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, vbo.live_vertex_confid);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::kMaxNumSurfels, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, vbo.live_normal_radius);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::kMaxNumSurfels, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, vbo.color_time);
	glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float) * Constants::kMaxNumSurfels, NULL, GL_DYNAMIC_DRAW);

	//Register buffer to cuda
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&(vbo.cuda_vbo_resources[0]), vbo.reference_vertex_confid, cudaGraphicsRegisterFlagsNone));
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&(vbo.cuda_vbo_resources[1]), vbo.reference_normal_radius, cudaGraphicsRegisterFlagsNone));
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&(vbo.cuda_vbo_resources[2]), vbo.live_vertex_confid, cudaGraphicsRegisterFlagsNone));
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&(vbo.cuda_vbo_resources[3]), vbo.live_normal_radius, cudaGraphicsRegisterFlagsNone));
	cudaSafeCall(cudaGraphicsGLRegisterBuffer(&(vbo.cuda_vbo_resources[4]), vbo.color_time, cudaGraphicsRegisterFlagsNone));
	cudaSafeCall(cudaGetLastError());
}

void surfelwarp::releaseGLSurfelGeometry(surfelwarp::GLSurfelGeometryVBO &vbo) {
	//First un-register the resource
	cudaSafeCall(cudaGraphicsUnregisterResource(vbo.cuda_vbo_resources[0]));
	cudaSafeCall(cudaGraphicsUnregisterResource(vbo.cuda_vbo_resources[1]));
	cudaSafeCall(cudaGraphicsUnregisterResource(vbo.cuda_vbo_resources[2]));
	cudaSafeCall(cudaGraphicsUnregisterResource(vbo.cuda_vbo_resources[3]));
	cudaSafeCall(cudaGraphicsUnregisterResource(vbo.cuda_vbo_resources[4]));

	//Then we can delete the vbos
	glDeleteBuffers(1, &(vbo.reference_vertex_confid));
	glDeleteBuffers(1, &(vbo.reference_normal_radius));
	glDeleteBuffers(1, &(vbo.live_vertex_confid));
	glDeleteBuffers(1, &(vbo.live_normal_radius));
	glDeleteBuffers(1, &(vbo.color_time));
}


