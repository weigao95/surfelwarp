//
// Created by wei on 3/18/18.
//

#include "common/Constants.h"
#include "common/ConfigParser.h"
#include "imgproc/ImageProcessor.h"
#include "imgproc/frameio/VolumeDeformFileFetch.h"
#include "core/SurfelGeometry.h"
#include "core/WarpField.h"
#include "core/render/Renderer.h"
#include "core/geometry/GeometryInitializer.h"
#include "core/geometry/VoxelSubsamplerSorting.h"
#include "visualization/Visualizer.h"
#include <iostream>
#include <thread>


int main() {
	using namespace surfelwarp;
	std::cout << "Test of rendering" << std::endl;
	
	//Parpare the test data
	auto parser = ConfigParser::Instance();
	
	//First test fetching
	FileFetch::Ptr fetcher = std::make_shared<FileFetch>(parser.data_path());
	ImageProcessor::Ptr processor = std::make_shared<ImageProcessor>(fetcher);
	
	//Test the frist frame case
	CameraObservation first_frame;
	processor->ProcessFirstFrameSerial(first_frame, 0);
	
	//Construct it
	Renderer renderer(parser.clip_image_rows(), parser.clip_image_cols());
	
	//The geometry class
	SurfelGeometry geometry;
	renderer.MapSurfelGeometryToCuda(1, geometry);
	renderer.UnmapSurfelGeometryFromCuda(1);
	renderer.MapSurfelGeometryToCuda(1);
	
	//The warp field
	WarpField warp_field;
	
	//The initializer
	GeometryInitializer::Ptr initializer = std::make_shared<GeometryInitializer>();
	initializer->AllocateBuffer();
	initializer->InitFromObservationSerial(geometry, first_frame);
	renderer.UnmapSurfelGeometryFromCuda(1);
	
	//Prepare for arguments
	auto geometry_attributes = geometry.Geometry();
	const auto num_vertex = geometry_attributes.live_vertex_confid.Size();
	const Matrix4f world2camera = Matrix4f::Identity();
	const auto current_time = 1.0f;
	
	//Draw the fusion maps
	//renderer.DrawFusionMaps(num_vertex, 1, world2camera);
	//renderer.DebugFusionMapsDraw(num_vertex, 1);
	
	//Draw the solver maps
	renderer.DrawSolverMapsWithRecentObservation(num_vertex, 1, current_time, world2camera);
	renderer.DebugSolverMapsDraw(num_vertex, 1);
	
	using FusionMaps = Renderer::FusionMaps;
	using SolverMaps = Renderer::SolverMaps;
	//FusionMaps maps;
	//renderer.MapFusionMapsToCuda(maps);
	SolverMaps maps;
	renderer.MapSolverMapsToCuda(maps);
	
	//Draw it, with new threads as opengl context are thread-wise
	auto draw_func = [&]()->void {
		//Visualizer::DrawColoredPointCloud(vertex, color_time);
		//Visualizer::DrawColorTimeMap(maps.color_time_map);
		//Visualizer::DrawPointCloudWithNormal(first_frame.vertex_config_map, first_frame.normal_radius_map);
		//Visualizer::DrawPointCloudWithNormal(maps.warp_vertex_map, maps.warp_normal_map);
		//Visualizer::DrawPointCloudWithNormal(maps.reference_vertex_map, maps.reference_normal_map);
		//Visualizer::DrawNormalMap(maps.warp_normal_map);
		//Visualizer::DrawNormalMap(first_frame.normal_radius_map);
		//Visualizer::DrawNormalizeRGBImage(maps.normalized_rgb_map);
		//Visualizer::DrawColorTimeMap(maps.normalized_rgb_map);
		
		Visualizer::DrawNormalizeRGBImage(maps.normalized_rgb_map);
	};
	
	std::thread draw_thread(draw_func);
	draw_thread.join();
	
	//renderer.UnmapFusionMapsFromCuda();
	renderer.UnmapSolverMapsFromCuda();
	
	std::cout << "Test done" << std::endl;
}
