//
// Created by wei on 3/18/18.
//

#pragma once

#include "core/render/glad/glad.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

//Cuda headers
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

//STL headers
#include <tuple>
#include <vector>
#include <Eigen/Eigen>
#include <memory>

//The type decals
#include "common/common_types.h"
#include "core/render/GLSurfelGeometryVBO.h"
#include "core/render/GLRenderedMaps.h"
#include "core/render/GLClearValues.h"
#include "core/render/GLShaderProgram.h"

namespace surfelwarp {
	
	
	class Renderer {
	private:
		//These member should be obtained from the config parser
		int m_image_width;
		int m_image_height;
		int m_fusion_map_width;
		int m_fusion_map_height;
		
		//The parameters that is accessed by drawing pipelines
		float4 m_renderer_intrinsic;
		float4 m_width_height_maxdepth;
	public:
		//Accessed by pointer
		using Ptr = std::shared_ptr<Renderer>;
		explicit Renderer(int image_rows, int image_cols);
		~Renderer();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(Renderer);
		
		
		/* GLFW windows related variables and functions
		 */
	private:
		GLFWmonitor* mGLFWmonitor = nullptr;
		GLFWwindow* mGLFWwindow = nullptr;
		void initGLFW();
		
		
		/* The buffer and method to clear the image
		 */
	private:
		GLClearValues m_clear_values;
		void initClearValues();
		
		
		/* The vertex buffer objects for surfel geometry
		 * Note that a double-buffer scheme is used here
		 */
	private:
		GLSurfelGeometryVBO m_surfel_geometry_vbos[2];
		void initVertexBufferObjects();
		void freeVertexBufferObjects();
	public:
		void MapSurfelGeometryToCuda(int idx, SurfelGeometry& geometry, cudaStream_t stream = 0);
		void MapSurfelGeometryToCuda(int idx, cudaStream_t stream = 0);
		void UnmapSurfelGeometryFromCuda(int idx, cudaStream_t stream = 0);
		

		/* The buffer for rendered maps
		 */
	private:
		//The frame/render buffer required for online processing
		GLFusionMapsFrameRenderBufferObjects m_fusion_map_buffers;
		GLSolverMapsFrameRenderBufferObjects m_solver_map_buffers;
		
		//The frame/render buffer for offline visualization
		GLOfflineVisualizationFrameRenderBufferObjects m_visualization_draw_buffers;
		void initFrameRenderBuffers();
		void freeFrameRenderBuffers();
		
		
		/* The vao for rendering, must be init after
		 * the initialization of vbos
		 */
	private:
		//The vao for processing, correspond to double buffer scheme
		GLuint m_fusion_map_vao[2];
		GLuint m_solver_map_vao[2];
		
		//The vao for offline visualization of reference and live geometry
		GLuint m_reference_geometry_vao[2];
		GLuint m_live_geometry_vao[2];
		void initMapRenderVAO();
		
		
		/* The shader program to render the maps for
		 * solver and geometry updater
		 */
	private:
		GLShaderProgram m_fusion_map_shader;
		GLShaderProgram m_solver_map_shader; //This shader will draw recent observation
		void initProcessingShaders();

		//the collect of shaders for visualization
		struct {
			GLShaderProgram normal_map;
			GLShaderProgram phong_map;
			GLShaderProgram albedo_map;
		} m_visualization_shaders;
		void initVisualizationShaders();
		void initShaders();
		
		//The workforce method for solver maps drawing
		void drawSolverMaps(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent_observation);

		//The workforce method for offline visualization
		void drawVisualizationMap(
			GLShaderProgram& shader, 
			GLuint geometry_vao, 
			unsigned num_vertex, int current_time, 
			const Matrix4f& world2camera, 
			bool with_recent_observation
		);

	public:
		void DrawFusionMaps(unsigned num_vertex, int vao_idx, const Matrix4f& world2camera);
		void DrawSolverMapsConfidentObservation(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera);
		void DrawSolverMapsWithRecentObservation(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera);

		//The offline visualization methods
		void SaveLiveNormalMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, const std::string& path, bool with_recent = true);
		void SaveLiveAlbedoMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, const std::string& path, bool with_recent = true);
		void SaveLivePhongMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, const std::string& path, bool with_recent = true);
		void SaveReferenceNormalMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, const std::string& path, bool with_recent = true);
		void SaveReferenceAlbedoMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, const std::string& path, bool with_recent = true);
		void SaveReferencePhongMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, const std::string& path, bool with_recent = true);
		
		//Debug drawing functions
		void DebugFusionMapsDraw(unsigned num_vertex, int vao_idx);
		void DebugSolverMapsDraw(unsigned num_vertex, int vao_idx);
		
		
		
		/* The access of fusion map
		 */
	public:
		struct FusionMaps {
			cudaTextureObject_t warp_vertex_map;
			cudaTextureObject_t warp_normal_map;
			cudaTextureObject_t index_map;
			cudaTextureObject_t color_time_map;
		};
		void MapFusionMapsToCuda(FusionMaps& maps, cudaStream_t stream = 0);
		void UnmapFusionMapsFromCuda(cudaStream_t stream = 0);
		
		/* The access of solver maps
		 */
	public:
		struct SolverMaps {
			cudaTextureObject_t reference_vertex_map;
			cudaTextureObject_t reference_normal_map;
			cudaTextureObject_t warp_vertex_map;
			cudaTextureObject_t warp_normal_map;
			cudaTextureObject_t index_map;
			cudaTextureObject_t normalized_rgb_map;
		};
		void MapSolverMapsToCuda(SolverMaps& maps, cudaStream_t stream = 0);
		void UnmapSolverMapsFromCuda(cudaStream_t stream = 0);
	};
	
} // namespace surfelwarp