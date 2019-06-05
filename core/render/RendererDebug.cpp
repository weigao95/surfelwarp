//
// Created by wei on 3/26/18.
//

#include "core/render/Renderer.h"
#include "math/vector_ops.hpp"
#include <opencv2/opencv.hpp>

void surfelwarp::Renderer::DebugFusionMapsDraw(unsigned num_vertex, int vao_idx)
{
	cv::Mat vertex_map(m_fusion_map_height, m_fusion_map_width, CV_32FC4);
	glBindFramebuffer(GL_FRAMEBUFFER, m_fusion_map_buffers.fusion_map_fbo);
	glReadPixels(0, 0, m_fusion_map_width, m_fusion_map_height, GL_RGBA, GL_FLOAT, vertex_map.data);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	//Looks stupid, but this only debug
	std::vector<float4> host_points; host_points.clear();
	DeviceArray2D<float4> canVertexMapDevice(m_fusion_map_height, m_fusion_map_width);
	canVertexMapDevice.upload(vertex_map.data, sizeof(float4)*m_fusion_map_width, m_fusion_map_height, m_fusion_map_width);
	canVertexMapDevice.download(host_points, m_fusion_map_width);
	
	//First count the number of points
	std::vector<float4> valid_points; valid_points.clear();
	for (size_t idx = 0; idx < host_points.size(); idx++) {
		float4 point = host_points[idx];
		if(std::abs(point.x > 1e-3) || std::abs(point.y > 1e-3) || std::abs(point.z > 1e-3)) {
			valid_points.push_back(point);
		}
	}
	
	LOG(INFO) << "The number of input vertex in fusion rendering is " << num_vertex;
	LOG(INFO) << "The number of valid points in fusion vertex map is " << valid_points.size();
	
	{
		//Also read the index value from cuda texture
		DeviceArray<int> index_array; index_array.create(m_fusion_map_height * m_fusion_map_width);
		cudaArray_t index_texture_ptr;
		cudaGraphicsMapResources(1, &(m_fusion_map_buffers.cuda_rbo_resources[2]));
		cudaGraphicsSubResourceGetMappedArray(&index_texture_ptr, m_fusion_map_buffers.cuda_rbo_resources[2], 0, 0);
		cudaMemcpyFromArray(index_array.ptr(), index_texture_ptr, 0, 0, index_array.sizeBytes(), cudaMemcpyDeviceToDevice);
		cudaGraphicsUnmapResources(1, &(m_fusion_map_buffers.cuda_rbo_resources[2]));
		cudaSafeCall(cudaGetLastError());
		std::vector<int> cpu_index;
		index_array.download(cpu_index);
		
		
		//Download the point for rendering
		int size_vec4 = num_vertex;
		int size_bytes = sizeof(float4) * size_vec4;
		std::vector<float4> cpu_vbo;
		cpu_vbo.resize(size_vec4);
		vao_idx %= 2;
		glBindBuffer(GL_ARRAY_BUFFER, m_surfel_geometry_vbos[vao_idx].live_vertex_confid);
		glGetBufferSubData(GL_ARRAY_BUFFER, 0, size_bytes, cpu_vbo.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		//The visited index
		std::vector<bool> visited_input;
		visited_input.resize(num_vertex);
		std::fill(visited_input.begin(), visited_input.end(), false);
		
		//Check the rendering
		for (int row_idx = 0; row_idx < m_fusion_map_height; row_idx++) {
			for (int col_idx = 0; col_idx < m_fusion_map_width; col_idx++) {
				int flat_idx = col_idx + row_idx * m_fusion_map_width;
				float4 point = host_points[flat_idx];
				int index = cpu_index[flat_idx];
				if (index != 0xFFFFFFFF) {
					float4 vbo_point = cpu_vbo[index];
					visited_input[index] = true;
					if (norm(vbo_point - point) > 1e-4) {
						LOG(ERROR) << "The element" << index << " is not matched";
					}
				}
			}
		}
		
		//Check which input is not visited
		std::vector<int> notvisited_index;
		std::vector<float4> notvisied_points;
		for(auto i = 0; i < visited_input.size(); i++) {
			if(!visited_input[i]) {
				notvisited_index.push_back(i);
				const auto point = cpu_vbo[i];
				notvisied_points.push_back(point);
				LOG(INFO) << "Model surfel " << i << " is not rendered in the map";
			}
		}
		
		//Project the non-visited points on map
		for(auto i = 0; i < notvisied_points.size(); i++) {
			const auto vertex = notvisied_points[i];
			const auto vertex_index = notvisited_index[i];
			int x = int((vertex.x / (vertex.z + 0e-10)) * 570) + 300;
			int y = int((vertex.y / (vertex.z + 0e-10)) * 570) + 220;
			LOG(INFO) << "The pixel position is " << x << " " << y;
			int flat_idx = x + y * m_fusion_map_width;
			float4 point = host_points[flat_idx];
			int index = cpu_index[flat_idx];
			if(index != 0xFFFFFFFF) {
				LOG(INFO) << "The difference is " << norm(vertex - point) << ", the diff on z is " << vertex.z - point.z;
			} else {
				LOG(INFO) << "For nonvisited vertex " << i << " is not found";
			}
		}
	} // end of index check
} // end of method

void surfelwarp::Renderer::DebugSolverMapsDraw(unsigned num_vertex, int vao_idx) {
	//First download the reference vertex map
	cv::Mat vertex_map(m_image_height, m_image_width, CV_32FC4);
	glBindFramebuffer(GL_FRAMEBUFFER, m_solver_map_buffers.solver_map_fbo);
	glReadPixels(0, 0, m_image_width, m_image_height, GL_RGBA, GL_FLOAT, vertex_map.data);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	//Looks stupid, but this only debug
	std::vector<float4> host_points; host_points.clear();
	DeviceArray2D<float4> canVertexMapDevice(m_image_height, m_image_width);
	canVertexMapDevice.upload(vertex_map.data, sizeof(float4)*m_image_width, m_image_height, m_image_width);
	canVertexMapDevice.download(host_points, m_image_width);
	
	//First count the number of points
	std::vector<float4> valid_points; valid_points.clear();
	for (size_t idx = 0; idx < host_points.size(); idx++) {
		float4 point = host_points[idx];
		if(std::abs(point.x > 1e-3) || std::abs(point.y > 1e-3) || std::abs(point.z > 1e-3)) {
			valid_points.push_back(point);
		}
	}
	
	LOG(INFO) << "The number of input vertex in solver map rendering is " << num_vertex;
	LOG(INFO) << "The number of valid points in solver reference vertex map is " << valid_points.size();
	
	//Test of index map
	{
		LOG(INFO) << "Check if index map in solver maps rendering";
		//Also read the index value from cuda texture
		DeviceArray<int> index_array; index_array.create(m_image_height * m_image_width);
		cudaArray_t index_texture_ptr;
		cudaGraphicsMapResources(1, &(m_solver_map_buffers.cuda_rbo_resources[4]));
		cudaGraphicsSubResourceGetMappedArray(&index_texture_ptr, m_solver_map_buffers.cuda_rbo_resources[4], 0, 0);
		cudaMemcpyFromArray(index_array.ptr(), index_texture_ptr, 0, 0, index_array.sizeBytes(), cudaMemcpyDeviceToDevice);
		cudaGraphicsUnmapResources(1, &(m_solver_map_buffers.cuda_rbo_resources[4]));
		cudaSafeCall(cudaGetLastError());
		std::vector<int> cpu_index;
		index_array.download(cpu_index);
		
		
		//Download the point for rendering
		int size_vec4 = num_vertex;
		int size_bytes = sizeof(float4) * size_vec4;
		std::vector<float4> cpu_vbo;
		cpu_vbo.resize(size_vec4);
		vao_idx %= 2;
		glBindBuffer(GL_ARRAY_BUFFER, m_surfel_geometry_vbos[vao_idx].live_vertex_confid);
		glGetBufferSubData(GL_ARRAY_BUFFER, 0, size_bytes, cpu_vbo.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		//The visited index
		std::vector<bool> visited_input;
		visited_input.resize(num_vertex);
		std::fill(visited_input.begin(), visited_input.end(), false);
		
		//Check the rendering
		for (int row_idx = 0; row_idx < m_image_height; row_idx++) {
			for (int col_idx = 0; col_idx < m_image_width; col_idx++) {
				int flat_idx = col_idx + row_idx * m_image_width;
				float4 point = host_points[flat_idx];
				int index = cpu_index[flat_idx];
				if (index != 0xFFFFFFFF) {
					float4 vbo_point = cpu_vbo[index];
					visited_input[index] = true;
					if (norm(vbo_point - point) > 1e-4) {
						LOG(ERROR) << "The element" << index << " is not matched";
					}
				}
			}
		}
		
		LOG(INFO) << "Solver index map check done";
	} // end of index check
}
