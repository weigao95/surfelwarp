//
// Created by wei on 3/18/18.
//

#include "common/common_types.h"
#include "common/logging.h"
#include "core/render/GLRenderedMaps.h"

#include <cuda_gl_interop.h>
#include <opencv2/opencv.hpp>

static const cudaResourceDesc& resource_desc_cuarray() {
	static cudaResourceDesc desc;
	memset(&desc, 0, sizeof(cudaResourceDesc));
	desc.resType = cudaResourceTypeArray;
	return desc;
}

static const cudaTextureDesc& texture_desc_default2d() {
	static cudaTextureDesc desc;
	memset(&desc, 0, sizeof(cudaTextureDesc));
	desc.addressMode[0] = cudaAddressModeBorder;
	desc.addressMode[1] = cudaAddressModeBorder;
	desc.filterMode = cudaFilterModePoint;
	desc.readMode = cudaReadModeElementType;
	desc.normalizedCoords = 0;
	return desc;
}

void surfelwarp::GLFusionMapsFrameRenderBufferObjects::initialize(
	int scaled_width, int scaled_height
) {
	//Generate the framebuffer object
	glGenFramebuffers(1, &fusion_map_fbo);
	
	//The render buffer for this frame
	glGenRenderbuffers(1, &warp_vertex_map);
	glGenRenderbuffers(1, &warp_normal_map);
	glGenRenderbuffers(1, &color_time_map);
	glGenRenderbuffers(1, &index_map);
	glGenRenderbuffers(1, &depth_buffer);
	
	//Allocate data storage for render buffer
	glBindRenderbuffer(GL_RENDERBUFFER, warp_vertex_map);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaled_width, scaled_height);
	glBindRenderbuffer(GL_RENDERBUFFER, warp_normal_map);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaled_width, scaled_height);
	glBindRenderbuffer(GL_RENDERBUFFER, color_time_map);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, scaled_width, scaled_height);
	glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, scaled_width, scaled_height);
	
	//The data in index map shall be unsigned int
	glBindRenderbuffer(GL_RENDERBUFFER, index_map);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_LUMINANCE32UI_EXT, scaled_width, scaled_height);
	
	//Attach the render buffer to framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, fusion_map_fbo);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, warp_vertex_map);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, warp_normal_map);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_RENDERBUFFER, index_map);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_RENDERBUFFER, color_time_map);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer);
	
	//Check the framebuffer attachment
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		LOG(FATAL) << "The frame buffer of surfel fusion is not complete";
	}
	
	//Enable draw-buffers
	GLuint draw_buffers[] = {
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		GL_COLOR_ATTACHMENT2,
		GL_COLOR_ATTACHMENT3
	};
	glDrawBuffers(4, draw_buffers);
	
	//Clean-up
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	//Initialize the cuda access to render buffer object
	cudaSafeCall(cudaGraphicsGLRegisterImage(&cuda_rbo_resources[0], warp_vertex_map, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&cuda_rbo_resources[1], warp_normal_map, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&cuda_rbo_resources[2], index_map, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&cuda_rbo_resources[3], color_time_map, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGetLastError());
}

void surfelwarp::GLFusionMapsFrameRenderBufferObjects::release() {
	//Release the cuda access to these buffers
	cudaSafeCall(cudaGraphicsUnregisterResource(cuda_rbo_resources[0]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cuda_rbo_resources[1]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cuda_rbo_resources[2]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cuda_rbo_resources[3]));

	//Release the render buffer objects and frame buffer objects
	glDeleteRenderbuffers(1, &warp_vertex_map);
	glDeleteRenderbuffers(1, &warp_normal_map);
	glDeleteRenderbuffers(1, &color_time_map);
	glDeleteRenderbuffers(1, &index_map);
	glDeleteRenderbuffers(1, &depth_buffer);
	
	//Clear the frame buffer objects
	glDeleteFramebuffers(1, &fusion_map_fbo);
}


void surfelwarp::GLFusionMapsFrameRenderBufferObjects::mapToCuda(
	cudaTextureObject_t & warp_vertex_texture, 
	cudaTextureObject_t & warp_normal_texture, 
	cudaTextureObject_t & index_texture, 
	cudaTextureObject_t & color_time_texture, 
	cudaStream_t stream
) {
	//First map the resource
	cudaSafeCall(cudaGraphicsMapResources(4, cuda_rbo_resources, stream));

	//The cudaArray
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&(cuda_mapped_arrays[0]), cuda_rbo_resources[0], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&(cuda_mapped_arrays[1]), cuda_rbo_resources[1], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&(cuda_mapped_arrays[2]), cuda_rbo_resources[2], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&(cuda_mapped_arrays[3]), cuda_rbo_resources[3], 0, 0));

	//Create texture
	cudaResourceDesc resource_desc = resource_desc_cuarray();
	cudaTextureDesc texture_desc = texture_desc_default2d();
	resource_desc.res.array.array = cuda_mapped_arrays[0];
	cudaCreateTextureObject(&(cuda_mapped_texture[0]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cuda_mapped_arrays[1];
	cudaCreateTextureObject(&(cuda_mapped_texture[1]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cuda_mapped_arrays[2];
	cudaCreateTextureObject(&(cuda_mapped_texture[2]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cuda_mapped_arrays[3];
	cudaCreateTextureObject(&(cuda_mapped_texture[3]), &resource_desc, &texture_desc, NULL);

	//Store the result
	warp_vertex_texture = cuda_mapped_texture[0];
	warp_normal_texture = cuda_mapped_texture[1];
	index_texture = cuda_mapped_texture[2];
	color_time_texture = cuda_mapped_texture[3];
}

void surfelwarp::GLFusionMapsFrameRenderBufferObjects::unmapFromCuda(cudaStream_t stream){
	cudaSafeCall(cudaGraphicsUnmapResources(4, cuda_rbo_resources, stream));
}

/* The method for solver maps
 */
void surfelwarp::GLSolverMapsFrameRenderBufferObjects::initialize(int width, int height) {
	//Generate the framebuffer object
	glGenFramebuffers(1, &solver_map_fbo);
	
	//The render buffer for this frame
	glGenRenderbuffers(1, &reference_vertex_map);
	glGenRenderbuffers(1, &reference_normal_map);
	glGenRenderbuffers(1, &warp_vertex_map);
	glGenRenderbuffers(1, &warp_normal_map);
	glGenRenderbuffers(1, &index_map);
	glGenRenderbuffers(1, &normalized_rgb_map);
	glGenRenderbuffers(1, &depth_buffer);
	
	//Allocate data storage for render buffer
	glBindRenderbuffer(GL_RENDERBUFFER, reference_vertex_map); //0
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, reference_normal_map); //1
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, warp_vertex_map);      //2
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, warp_normal_map);      //3
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, normalized_rgb_map);   //4
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);         //5
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);
	
	//The data in index map shall be unsigned int
	glBindRenderbuffer(GL_RENDERBUFFER, index_map);            //6
	glRenderbufferStorage(GL_RENDERBUFFER, GL_LUMINANCE32UI_EXT, width, height);
	
	//Attach the render buffer to framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, solver_map_fbo);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, reference_vertex_map);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, reference_normal_map);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_RENDERBUFFER, warp_vertex_map);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_RENDERBUFFER, warp_normal_map);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_RENDERBUFFER, index_map);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, GL_RENDERBUFFER, normalized_rgb_map);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer);
	
	//Check the framebuffer attachment
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		LOG(FATAL) << "The frame buffer of point solver is not complete" ;
	}
	
	//Enable draw-buffers
	GLuint draw_buffers[] = {
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		GL_COLOR_ATTACHMENT2,
		GL_COLOR_ATTACHMENT3,
		GL_COLOR_ATTACHMENT4,
		GL_COLOR_ATTACHMENT5
	};
	glDrawBuffers(6, draw_buffers);
	
	//Clean-up
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	//Initialize the cuda resource
	cudaSafeCall(cudaGraphicsGLRegisterImage(&cuda_rbo_resources[0], reference_vertex_map, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&cuda_rbo_resources[1], reference_normal_map, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&cuda_rbo_resources[2], warp_vertex_map, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&cuda_rbo_resources[3], warp_normal_map, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&cuda_rbo_resources[4], index_map, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGraphicsGLRegisterImage(&cuda_rbo_resources[5], normalized_rgb_map, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsReadOnly));
	cudaSafeCall(cudaGetLastError());
}

void surfelwarp::GLSolverMapsFrameRenderBufferObjects::release() {
	//Release the resource by cuda
	cudaSafeCall(cudaGraphicsUnregisterResource(cuda_rbo_resources[0]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cuda_rbo_resources[1]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cuda_rbo_resources[2]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cuda_rbo_resources[3]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cuda_rbo_resources[4]));
	cudaSafeCall(cudaGraphicsUnregisterResource(cuda_rbo_resources[5]));
	
	//Now we can release the buffer
	glDeleteRenderbuffers(1, &reference_vertex_map);
	glDeleteRenderbuffers(1, &reference_normal_map);
	glDeleteRenderbuffers(1, &index_map);
	glDeleteRenderbuffers(1, &warp_vertex_map);
	glDeleteRenderbuffers(1, &warp_normal_map);
	glDeleteRenderbuffers(1, &normalized_rgb_map);
	glDeleteRenderbuffers(1, &depth_buffer);
	
	glDeleteFramebuffers(1, &solver_map_fbo);
}

void surfelwarp::GLSolverMapsFrameRenderBufferObjects::mapToCuda(
	cudaTextureObject_t &reference_vertex_texture,
	cudaTextureObject_t &reference_normal_texture,
	cudaTextureObject_t &warp_vertex_texture,
	cudaTextureObject_t &warp_normal_texture,
	cudaTextureObject_t &index_texture,
	cudaTextureObject_t &normalized_rgb_texture,
	cudaStream_t stream
) {
	//First map the resource
	cudaSafeCall(cudaGraphicsMapResources(6, cuda_rbo_resources, stream));
	
	//The cudaArray
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&(cuda_mapped_arrays[0]), cuda_rbo_resources[0], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&(cuda_mapped_arrays[1]), cuda_rbo_resources[1], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&(cuda_mapped_arrays[2]), cuda_rbo_resources[2], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&(cuda_mapped_arrays[3]), cuda_rbo_resources[3], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&(cuda_mapped_arrays[4]), cuda_rbo_resources[4], 0, 0));
	cudaSafeCall(cudaGraphicsSubResourceGetMappedArray(&(cuda_mapped_arrays[5]), cuda_rbo_resources[5], 0, 0));
	
	//Create texture
	cudaResourceDesc resource_desc = resource_desc_cuarray();
	cudaTextureDesc texture_desc = texture_desc_default2d();
	resource_desc.res.array.array = cuda_mapped_arrays[0];
	cudaCreateTextureObject(&(cuda_mapped_texture[0]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cuda_mapped_arrays[1];
	cudaCreateTextureObject(&(cuda_mapped_texture[1]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cuda_mapped_arrays[2];
	cudaCreateTextureObject(&(cuda_mapped_texture[2]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cuda_mapped_arrays[3];
	cudaCreateTextureObject(&(cuda_mapped_texture[3]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cuda_mapped_arrays[4];
	cudaCreateTextureObject(&(cuda_mapped_texture[4]), &resource_desc, &texture_desc, NULL);
	resource_desc.res.array.array = cuda_mapped_arrays[5];
	cudaCreateTextureObject(&(cuda_mapped_texture[5]), &resource_desc, &texture_desc, NULL);
	
	//Store the result
	reference_vertex_texture = cuda_mapped_texture[0];
	reference_normal_texture = cuda_mapped_texture[1];
	warp_vertex_texture = cuda_mapped_texture[2];
	warp_normal_texture = cuda_mapped_texture[3];
	index_texture = cuda_mapped_texture[4];
	normalized_rgb_texture = cuda_mapped_texture[5];
}

void surfelwarp::GLSolverMapsFrameRenderBufferObjects::unmapFromCuda(cudaStream_t stream) {
	cudaSafeCall(cudaGraphicsUnmapResources(6, cuda_rbo_resources, stream));
}


/* The buffer and methods for offline visualization frame & render buffer objects
 */
void surfelwarp::GLOfflineVisualizationFrameRenderBufferObjects::initialize(int width, int height) {
	//Generate the framebuffer object
	glGenFramebuffers(1, &visualization_map_fbo);
	
	//The render buffer for this frame
	glGenRenderbuffers(1, &normalized_rgba_rbo);
	glGenRenderbuffers(1, &depth_buffer);
	
	//Allocate data storage for render buffer
	glBindRenderbuffer(GL_RENDERBUFFER, normalized_rgba_rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, width, height);
	
	//Attach the render buffer to framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, visualization_map_fbo);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, normalized_rgba_rbo);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer);
	
	//Check the framebuffer attachment
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		LOG(FATAL) << "The frame buffer for visualization is not complete" ;
	}
	
	//Enable draw-buffers
	GLuint draw_buffers[] = {
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1
	};
	glDrawBuffers(1, draw_buffers);
	
	//Clean-up
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void surfelwarp::GLOfflineVisualizationFrameRenderBufferObjects::release() {
	glDeleteRenderbuffers(1, &normalized_rgba_rbo);
	glDeleteRenderbuffers(1, &depth_buffer);
	
	glDeleteFramebuffers(1, &visualization_map_fbo);
}



void surfelwarp::GLOfflineVisualizationFrameRenderBufferObjects::save(const std::string &path) {
	//Bind the render buffer object
	glBindRenderbuffer(GL_RENDERBUFFER, normalized_rgba_rbo);
	
	//First query the size of render buffer object
	GLint width, height;
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &width);
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &height);
	
	//Construct the storage
	cv::Mat rendered_map_cv(height, width, CV_8UC4);
	glBindFramebuffer(GL_FRAMEBUFFER, visualization_map_fbo);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rendered_map_cv.data);
	
	//Cleanup code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	
	//Save it
	cv::imwrite(path, rendered_map_cv);
}



