
#include "imgproc/frameio/AzureKinectDKFetch.h"
#include "visualization/Visualizer.h"
#include "imgproc/ImageProcessor.h"

using namespace surfelwarp;

#define WAIT_TIME 5

int main(int argc, char** argv){

	std::string config_path;
    
    if (argc <= 1) {
        config_path = "/data/surfelwarp/test_data/kinectdk_config.json";
	}else{
        config_path = std::string(argv[1]);
    }
	//Parse it
	auto& config = ConfigParser::Instance();
	config.ParseConfig(config_path);


    Visualizer::Ptr visualizer = std::make_shared<Visualizer>();

    FetchInterface::Ptr image_fetcher = std::make_shared<AzureKinectDKFetch>(config.data_path(), config.isSaveOnlineFrame());
    //ImageProcessor::Ptr image_processor = std::make_shared<ImageProcessor>(image_fetcher);

    cv::Mat d_img = cv::Mat(cv::Size(config.raw_image_cols(), config.raw_image_rows()), CV_16UC1);
	cv::Mat c_img = cv::Mat(cv::Size(config.raw_image_cols(), config.raw_image_rows()), CV_8UC3);

    //cv::Mat depth_img, rgb_img, prev_rgb_img;
    for (auto frame_idx = config.start_frame_idx(); frame_idx < config.num_frames(); frame_idx++){

        image_fetcher->FetchDepthImage(frame_idx, d_img);
        image_fetcher->FetchRGBImage(frame_idx, c_img);
        //image_processor->FetchFrame(frame_idx);
        visualizer->DrawDepthImage(d_img);
        cv::imshow("rgb", c_img);
        //cv::imshow("rgb_prev", image_processor->RawRGBImagePrevCPU());
        std::cout << "record: " << frame_idx << std::endl;
        int res = cv::waitKey(WAIT_TIME);
        if (res == 27 || res == 'q')
        {
            break;
        }
    }

    return 0;
}
