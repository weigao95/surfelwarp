//
// Created by wei on 3/18/18.
//

#pragma once
#include "core/render/glad/glad.h"
#include <cuda_runtime_api.h>

namespace surfelwarp {
	
	
	/**
	 * \brief The frame buffer object (FBO) and render buffer object (RBO)
	 *        used for point fusion. This class handles the mapping from/to cuda
	 */
	struct GLFusionMapsFrameRenderBufferObjects {
		GLuint fusion_map_fbo;
		
		//These are render buffer objects
		GLuint warp_vertex_map;
		GLuint warp_normal_map;
		GLuint index_map;
		GLuint color_time_map;
		GLuint depth_buffer;
		
		//The buffer for cuda access
		cudaGraphicsResource_t cuda_rbo_resources[4];
		cudaArray_t cuda_mapped_arrays[4];
		cudaTextureObject_t cuda_mapped_texture[4];
		
		//Can only be called by renderer
		void initialize(int scaled_width, int scaled_height);
		void release();

		//Map the resource to cuda, must be called after initialize
		void mapToCuda(
			cudaTextureObject_t& warp_vertex_texture, 
			cudaTextureObject_t& warp_normal_texture, 
			cudaTextureObject_t& index_texture, 
			cudaTextureObject_t& color_time_texture, 
			cudaStream_t stream = 0
		);
		void unmapFromCuda(cudaStream_t stream = 0);
	};
	
	
	/**
	 * \brief The FBO and RBO for solver maps. These buffer will also be
	 *        mapped to cuda access.
	 */
	struct GLSolverMapsFrameRenderBufferObjects {
		GLuint solver_map_fbo;
		
		//These are render buffer objects
		GLuint reference_vertex_map;
		GLuint reference_normal_map;
		GLuint warp_vertex_map;
		GLuint warp_normal_map;
		GLuint index_map;
		GLuint normalized_rgb_map;
		GLuint depth_buffer;
		
		//The resource for cuda access
		cudaGraphicsResource_t  cuda_rbo_resources[6];
		cudaArray_t cuda_mapped_arrays[6];
		cudaTextureObject_t cuda_mapped_texture[6];
		
		//Can only be accessed by renderer
		void initialize(int width, int height);
		void release();
		
		//Communicate the resource from/to cuda
		void mapToCuda(
			cudaTextureObject_t& reference_vertex_texture,
			cudaTextureObject_t& reference_normal_texture,
			cudaTextureObject_t& warp_vertex_texture,
			cudaTextureObject_t& warp_normal_texture,
			cudaTextureObject_t& index_texture,
			cudaTextureObject_t& normalized_rgb_texture,
			cudaStream_t stream = 0
		);
		void unmapFromCuda(cudaStream_t stream = 0);
	};
	
	
	/* The offline rendering map for float4 output. The buffer do not need to be
	 * mapped to cuda, but may need to be save offline
	 */
	struct GLOfflineVisualizationFrameRenderBufferObjects {
		GLuint visualization_map_fbo;
		GLuint normalized_rgba_rbo; //A float4 texture whose elements are in [0, 1]
		GLuint depth_buffer;
		
		//Construct and descruct the GPU memory
		void initialize(int width, int height);
		void release();
		
		//Save it as opencv images.
		//Of course, this method should never be used in real-time code
		void save(const std::string& path);
	};
}
