#include "common/common_types.h"
#include "common/sanity_check.h"
#include "common/ConfigParser.h"
#include "common/CameraObservation.h"
#include "visualization/Visualizer.h"
#include "imgproc/frameio/VolumeDeformFileFetch.h"
#include "imgproc/ImageProcessor.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>


void testBoundary()
{
	using namespace surfelwarp;
	//Parpare the test data
	auto parser = ConfigParser::Instance();

	//First test fetching
	FileFetch::Ptr fetcher = std::make_shared<FileFetch>(parser.data_path());
	ImageProcessor::Ptr processor = std::make_shared<ImageProcessor>(fetcher);
	
	//CameraObservation observation;
	//processor->ProcessFrameSerial(observation, 130);
	//Visualizer::DrawColoredPointCloud(observation.vertex_config_map, observation.color_time_map);

	//Process using the observation interface
	for(auto frame_idx = 10; frame_idx < 160; frame_idx++) {
		LOG(INFO) << "Current frame is " << frame_idx;
		CameraObservation observation;
		processor->ProcessFrameSerial(observation, frame_idx);
		if(frame_idx == 130) {
			Visualizer::DrawColoredPointCloud(observation.vertex_config_map, observation.color_time_map);
		}
	}
}

void testFullProcessing() {
	using namespace surfelwarp;
	//The path for config
	boost::filesystem::path config_path_prefix;
#if defined(WIN32)
	config_path_prefix = boost::filesystem::path("C:/Users/wei/Documents/Visual Studio 2015/Projects/surfelwarp/data/configs");
#else
	config_path_prefix = boost::filesystem::path("/home/wei/Documents/programs/surfelwarp/data/configs");
#endif

	//Parse it
	boost::filesystem::path config_path = config_path_prefix / "boxing_config.json";
	auto& parser = ConfigParser::Instance();
	parser.ParseConfig(config_path.string());

	
	//First test fetching
	FileFetch::Ptr fetcher = std::make_shared<FileFetch>(parser.data_path());
	ImageProcessor::Ptr processor = std::make_shared<ImageProcessor>(fetcher);
	
	//Do it
	CameraObservation observation;
	processor->ProcessFrameSerial(observation, parser.start_frame_idx() + 150);
	
	//Draw it
	auto draw_func = [&]() {
		//Visualizer::DrawPointCloud(observation.vertex_config_map);
		Visualizer::DrawSegmentMask(observation.foreground_mask, observation.normalized_rgba_map, 1);
		//Visualizer::DrawGrayScaleImage(observation.filter_foreground_mask);
	};
	
	//Use thread to draw it
	std::thread draw_thread(draw_func);
	draw_thread.join();
}

int main() {
	testFullProcessing();
}