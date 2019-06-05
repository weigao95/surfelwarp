//
// Created by wei on 3/26/18.
//

#include "core/render/Renderer.h"
#include "common/ConfigParser.h"
#include "common/Constants.h"
#include "core/warp_solver/solver_constants.h"

/* Compile and initialize the shader program
 * used for later drawing methods
 */
void surfelwarp::Renderer::initProcessingShaders() {
	//Query the shader path
	boost::filesystem::path file_path(__FILE__);
	const boost::filesystem::path render_path = file_path.parent_path();
	const boost::filesystem::path shader_path = render_path / "shaders";

	//Compile the shader for fusion map
	const auto fusion_map_vert_path = shader_path / "fusion_map.vert";
	const auto fusion_map_frag_path = shader_path / "fusion_map.frag";
	m_fusion_map_shader.Compile(fusion_map_vert_path.string(), fusion_map_frag_path.string());
	
	//Compile the shader for solver maps
	const auto solver_map_frag_path = shader_path / "solver_map.frag";

	//Use dense solver maps or not
#if defined(USE_DENSE_SOLVER_MAPS)
	const auto solver_map_recent_vert_path = shader_path / "solver_map_sized.vert";
#else
	const auto solver_map_recent_vert_path = shader_path / "solver_map.vert";
#endif
	m_solver_map_shader.Compile(solver_map_recent_vert_path.string(), solver_map_frag_path.string());
}

void surfelwarp::Renderer::initVisualizationShaders() {
	boost::filesystem::path file_path(__FILE__);
	const boost::filesystem::path render_path = file_path.parent_path();
	const boost::filesystem::path shader_path = render_path / "shaders";
	
	//The fragment shader for normal map, phong shading and albedo color
	const auto normal_map_frag_path = shader_path / "normal_as_color.frag";
	const auto phong_shading_path = shader_path / "phong_color.frag";
	const auto albedo_color_path = shader_path / "albedo_color.frag";
	
	//The vertex shader for referenc and live geometry
	const auto geometry_vert_path = shader_path / "geometry.vert";
	
	//Compile the shader without recent observation
	m_visualization_shaders.normal_map.Compile(geometry_vert_path.string(), normal_map_frag_path.string());
	m_visualization_shaders.phong_map.Compile(geometry_vert_path.string(), phong_shading_path.string());
	m_visualization_shaders.albedo_map.Compile(geometry_vert_path.string(), albedo_color_path.string());
}

void surfelwarp::Renderer::initShaders() {
	//The shader for warp solver, data fusion and visualization
	initProcessingShaders();
	initVisualizationShaders();
}


/* The drawing and debug function for fusion map
 */
void surfelwarp::Renderer::DrawFusionMaps(
	unsigned num_vertex,
	int vao_idx,
	const Matrix4f& world2camera
) {
	//Bind the shader
	m_fusion_map_shader.Bind();
	
	//The vao/vbo for the rendering
	vao_idx = vao_idx % 2;
	glBindVertexArray(m_fusion_map_vao[vao_idx]);
	
	//The framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, m_fusion_map_buffers.fusion_map_fbo);
	glViewport(0, 0, m_fusion_map_width, m_fusion_map_height);
	
	//Clear the render buffer object
	glClearBufferfv(GL_COLOR, 0, m_clear_values.vertex_map_clear);
	glClearBufferfv(GL_COLOR, 1, m_clear_values.normal_map_clear);
	glClearBufferuiv(GL_COLOR, 2, &(m_clear_values.index_map_clear));
	glClearBufferfv(GL_COLOR, 3, m_clear_values.color_time_clear);
	glClearBufferfv(GL_DEPTH, 0, &(m_clear_values.z_buffer_clear));
	
	//Set the uniform values
	m_fusion_map_shader.SetUniformMatrix("world2camera", world2camera);
	m_fusion_map_shader.SetUniformVector("intrinsic", m_renderer_intrinsic);
	m_fusion_map_shader.SetUniformVector("width_height_maxdepth", m_width_height_maxdepth);
	
	//Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);
	
	//Cleanup-code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindVertexArray(0);
	m_fusion_map_shader.Unbind();
}

/* The method to draw solver maps
 */
void surfelwarp::Renderer::drawSolverMaps(
	unsigned num_vertex,
	int vao_idx,
	int current_time,
	const surfelwarp::Matrix4f &world2camera,
	bool with_recent_observation
) {
	//Bind the shader
	m_solver_map_shader.Bind();
	
	//Normalize the vao index
	vao_idx %= 2;
	glBindVertexArray(m_solver_map_vao[vao_idx]);
	
	//The size is image rows/cols
	glBindFramebuffer(GL_FRAMEBUFFER, m_solver_map_buffers.solver_map_fbo);
	glViewport(0, 0, m_image_width, m_image_height);
	
	//Clear the render buffer object
	glClearBufferfv(GL_COLOR, 0, m_clear_values.vertex_map_clear);
	glClearBufferfv(GL_COLOR, 1, m_clear_values.normal_map_clear);
	glClearBufferfv(GL_COLOR, 2, m_clear_values.vertex_map_clear);
	glClearBufferfv(GL_COLOR, 3, m_clear_values.normal_map_clear);
	glClearBufferuiv(GL_COLOR, 4, &(m_clear_values.index_map_clear));
	glClearBufferfv(GL_COLOR, 5, m_clear_values.solver_rgba_clear);
	glClearBufferfv(GL_DEPTH, 0, &(m_clear_values.z_buffer_clear));
	
	//Set uniform values
	m_solver_map_shader.SetUniformMatrix("world2camera", world2camera);
	m_solver_map_shader.SetUniformVector("intrinsic", m_renderer_intrinsic);
	
	//The current time of the solver maps
	float4 width_height_maxdepth_currtime = make_float4(m_width_height_maxdepth.x, m_width_height_maxdepth.y, m_width_height_maxdepth.z, current_time);
	m_solver_map_shader.SetUniformVector("width_height_maxdepth_currtime", width_height_maxdepth_currtime);
	
	//The time threshold depend on input
	float2 confid_time_threshold = make_float2(Constants::kStableSurfelConfidenceThreshold, Constants::kRenderingRecentTimeThreshold);
	if(!with_recent_observation) {
		confid_time_threshold.y = -1.0f; //Do not pass any surfel due to recent observed
	}
	
	m_solver_map_shader.SetUniformVector("confid_time_threshold", confid_time_threshold);
	
	//Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);
	
	//Cleanup-code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindVertexArray(0);
	m_solver_map_shader.Unbind();
}

void surfelwarp::Renderer::DrawSolverMapsConfidentObservation(
	unsigned num_vertex,
	int vao_idx,
	int current_time, // Not used
	const surfelwarp::Matrix4f &world2camera
) {
	drawSolverMaps(num_vertex, vao_idx, current_time, world2camera, false);
}


void surfelwarp::Renderer::DrawSolverMapsWithRecentObservation(
	unsigned num_vertex,
	int vao_idx,
	int current_time,
	const surfelwarp::Matrix4f &world2camera
) {
	drawSolverMaps(num_vertex, vao_idx, current_time, world2camera, true);
}


/* The method for visualization map drawing
 */
void surfelwarp::Renderer::drawVisualizationMap(
	GLShaderProgram & shader, 
	GLuint geometry_vao, 
	unsigned num_vertex, int current_time, 
	const Matrix4f & world2camera, 
	bool with_recent_observation
) {
	//Bind the shader
	shader.Bind();

	//Use the provided vao
	glBindVertexArray(geometry_vao);

	//The size is image rows/cols
	glBindFramebuffer(GL_FRAMEBUFFER, m_visualization_draw_buffers.visualization_map_fbo);
	glViewport(0, 0, m_image_width, m_image_height);

	//Clear the render buffer object
	glClearBufferfv(GL_COLOR, 0, m_clear_values.visualize_rgba_clear);
	glClearBufferfv(GL_DEPTH, 0, &(m_clear_values.z_buffer_clear));

	//Set uniform values
	shader.SetUniformMatrix("world2camera", world2camera);
	shader.SetUniformVector("intrinsic", m_renderer_intrinsic);
	
	//The current time of the solver maps
	const float4 width_height_maxdepth_currtime = make_float4(m_width_height_maxdepth.x, m_width_height_maxdepth.y, m_width_height_maxdepth.z, current_time);
	shader.SetUniformVector("width_height_maxdepth_currtime", width_height_maxdepth_currtime);
	
	//The time threshold depend on input
	float2 confid_time_threshold = make_float2(Constants::kStableSurfelConfidenceThreshold, Constants::kRenderingRecentTimeThreshold);
	if(!with_recent_observation) {
		confid_time_threshold.y = -1.0f; //Do not pass any surfel due to recent observed
	}

	//Hand in to shader
	shader.SetUniformVector("confid_time_threshold", confid_time_threshold);

	//Draw it
	glDrawArrays(GL_POINTS, 0, num_vertex);

	//Cleanup code
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindVertexArray(0);
	shader.Unbind();
}

void surfelwarp::Renderer::SaveLiveNormalMap(
	unsigned num_vertex,
	int vao_idx,
	int current_time,
	const surfelwarp::Matrix4f &world2camera,
	const std::string &path,
	bool with_recent
) {
	//Draw it
	const auto geometry_vao = m_live_geometry_vao[vao_idx];
	drawVisualizationMap(
		m_visualization_shaders.normal_map,
		geometry_vao,
		num_vertex, current_time,
		world2camera,
		with_recent
	);
	
	//Save it
	m_visualization_draw_buffers.save(path);
}

void surfelwarp::Renderer::SaveLiveAlbedoMap(
	unsigned num_vertex,
	int vao_idx,
	int current_time,
	const surfelwarp::Matrix4f &world2camera,
	const std::string &path,
	bool with_recent
) {
	//Draw it
	const auto geometry_vao = m_live_geometry_vao[vao_idx];
	drawVisualizationMap(
		m_visualization_shaders.albedo_map,
		geometry_vao,
		num_vertex, current_time,
		world2camera,
		with_recent
	);
	
	//Save it
	m_visualization_draw_buffers.save(path);
}


void surfelwarp::Renderer::SaveLivePhongMap(
	unsigned num_vertex,
	int vao_idx,
	int current_time,
	const surfelwarp::Matrix4f &world2camera,
	const std::string &path,
	bool with_recent
) {
	//Draw it
	const auto geometry_vao = m_live_geometry_vao[vao_idx];
	drawVisualizationMap(
		m_visualization_shaders.phong_map,
		geometry_vao,
		num_vertex, current_time,
		world2camera,
		with_recent
	);
	
	//Save it
	m_visualization_draw_buffers.save(path);
}

void surfelwarp::Renderer::SaveReferenceNormalMap(
	unsigned num_vertex,
	int vao_idx, 
	int current_time, 
	const Matrix4f & world2camera, 
	const std::string & path, 
	bool with_recent
) {
	const auto geometry_vao = m_reference_geometry_vao[vao_idx];
	drawVisualizationMap(
		m_visualization_shaders.normal_map,
		geometry_vao,
		num_vertex, current_time,
		world2camera,
		with_recent
	);
	
	//Save it
	m_visualization_draw_buffers.save(path);
}

void surfelwarp::Renderer::SaveReferenceAlbedoMap(
	unsigned num_vertex, 
	int vao_idx, 
	int current_time, 
	const Matrix4f & world2camera, 
	const std::string & path, 
	bool with_recent
) {
	const auto geometry_vao = m_reference_geometry_vao[vao_idx];
	drawVisualizationMap(
		m_visualization_shaders.albedo_map,
		geometry_vao,
		num_vertex, current_time,
		world2camera,
		with_recent
	);
	
	//Save it
	m_visualization_draw_buffers.save(path);
}


void surfelwarp::Renderer::SaveReferencePhongMap(
	unsigned num_vertex, 
	int vao_idx, 
	int current_time, 
	const Matrix4f & world2camera, 
	const std::string & path, 
	bool with_recent
) {
	const auto geometry_vao = m_reference_geometry_vao[vao_idx];
	drawVisualizationMap(
		m_visualization_shaders.phong_map,
		geometry_vao,
		num_vertex, current_time,
		world2camera,
		with_recent
	);
	
	//Save it
	m_visualization_draw_buffers.save(path);
}