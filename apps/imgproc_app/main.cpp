//
// Created by wei on 5/29/18.
//
#include "common/ConfigParser.h"
#include "imgproc/frameio/VolumeDeformFileFetch.h"
#include "imgproc/ImageProcessor.h"
#include "visualization/Visualizer.h"

void saveProcessedImage(const surfelwarp::CameraObservation& observation, const boost::filesystem::path& data_dir, int frame_idx) {
	using namespace surfelwarp;
	
	//The foreground mask for visualization
	const std::string mask_name = "foreground_" + std::to_string(frame_idx) + ".png";
	Visualizer::SaveSegmentMask(observation.foreground_mask, observation.normalized_rgba_map, (data_dir / mask_name).string(), 1);
	
	//The raw mask
	char frame_idx_str[20];
	sprintf(frame_idx_str, "%06d", frame_idx);
	std::string raw_mask_name = "frame-";
	raw_mask_name += std::string(frame_idx_str);
	raw_mask_name += ".mask.png";
	Visualizer::SaveRawSegmentMask(observation.foreground_mask, (data_dir / raw_mask_name).string());
}


int main() {
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
	auto& config = ConfigParser::Instance();
	config.ParseConfig(config_path.string());
	
	
	const auto start_frame_idx = config.start_frame_idx();
	
	//Create the data dir
	boost::filesystem::path data_dir("result");
	if(!boost::filesystem::exists(data_dir)) {
		boost::filesystem::create_directory(data_dir);
	}
	
	//First test fetching
	FileFetch::Ptr fetcher = std::make_shared<FileFetch>(config.data_path());
	ImageProcessor::Ptr processor = std::make_shared<ImageProcessor>(fetcher);
	
	//From the first frame
	for(auto i = 1; i < 301; i++) {
		LOG(INFO) << "The frame " << i;
		CameraObservation observation;
		//processor->ProcessFrameSerial(observation, i + start_frame_idx);
		processor->ProcessFrameStreamed(observation, i + start_frame_idx);
		//saveProcessedImage(observation, data_dir, i + start_frame_idx);
		//Visualizer::DrawRawSegmentMask(observation.foreground_mask);
	}
}