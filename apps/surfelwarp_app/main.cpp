//
// Created by wei on 5/22/18.
//

#include "common/common_utils.h"
#include "common/ConfigParser.h"
#include "core/SurfelWarpSerial.h"
#include <boost/filesystem.hpp>

int main(int argc, char** argv) {
	using namespace surfelwarp;
	
	//Get the config path
	std::string config_path;
	if (argc <= 1) {
#if defined(WIN32)
		config_path = "C:/Users/wei/Documents/Visual Studio 2015/Projects/surfelwarp/test_data/boxing_config.json";
#else
		config_path = "/home/wei/Documents/programs/surfelwarp/test_data/boxing_config.json";
#endif
	} else {
		config_path = std::string(argv[1]);
	}

	//Parse it
	auto& config = ConfigParser::Instance();
	config.ParseConfig(config_path);

	//The context
	//auto context = initCudaContext();

	//Save offline
	bool offline_rendering = true;

	//The processing loop
	SurfelWarpSerial fusion;

	fusion.ProcessFirstFrame();
	for(auto i = 0; i < config.num_frames(); i++){
		LOG(INFO) << "The " << i << "th Frame";
		fusion.ProcessNextFrameWithReinit(offline_rendering);
	}
	
	//destroyCudaContext(context);
}
