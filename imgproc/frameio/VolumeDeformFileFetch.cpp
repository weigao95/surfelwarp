#include <opencv2/opencv.hpp>
#include <unistd.h>
#include "common/OpenCV_CrossPlatform.h"
#include "common/ConfigParser.h"
#include "common/logging.h"
#include "VolumeDeformFileFetch.h"


surfelwarp::VolumeDeformFileFetch::VolumeDeformFileFetch(const path& data_path) : m_data_path(data_path) {
	const auto & config = ConfigParser::Instance();
	// before read any image, m_cur_frame_idx = start_idx - 1
	m_cur_frame_idx = config.start_frame_idx() - 1;
	m_image_rows = config.raw_image_rows();
	m_image_cols = config.raw_image_cols();
    m_thread = std::thread(&surfelwarp::VolumeDeformFileFetch::ReadImageFromFile, this);
}



void surfelwarp::VolumeDeformFileFetch::FetchDepthImage(size_t frame_idx, cv::Mat & depth_img)
{
	//Read the image
	LOG(WARNING) << "Deprecated, please use FetchDepthAndRGBImage instaed!" << std::endl;
	if (frame_idx == m_cur_frame_idx){
		std::unique_lock<std::mutex> lock(m_mutex);
		depth_img = m_depth_images.front();
		m_depth_images.pop_front();
	}else{
		std::cout << "this should not happen, just for debug" << std::endl;
		std::cout << "fetch frame_idx: " << frame_idx << ";  cur_frame_idx: " << m_cur_frame_idx << std::endl;
	}
}

void surfelwarp::VolumeDeformFileFetch::FetchDepthImage(size_t frame_idx, void * depth_img)
{

}

void surfelwarp::VolumeDeformFileFetch::FetchRGBImage(size_t frame_idx, cv::Mat & rgb_img)
{
	LOG(WARNING) << "Deprecated, please use FetchDepthAndRGBImage instaed!" << std::endl;
	if(frame_idx == m_cur_frame_idx){
		std::unique_lock<std::mutex> lock(m_mutex);
		rgb_img = m_color_images.front();
		m_color_images.pop_front();
	}else{
		std::cout << "this should not happen, just for debug" << std::endl;
		std::cout << "fetch frame_idx: " << frame_idx << ";  cur_frame_idx: " << m_cur_frame_idx << std::endl;
	}
}

void surfelwarp::VolumeDeformFileFetch::FetchDepthAndRGBImage(size_t frame_idx, cv::Mat & depth_img, cv::Mat & rgb_img){
	// (xt): This loop maybe not needed, just leave it at now.
	for(int i = 0; i <= 3; i++){
		bool is_empty = true;
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			// make sure image has already been read 
			if(m_color_images.size() > 0 && m_color_images.size() > 0){
				is_empty = false;
				// make sure the current frame is the one wanted
				SURFELWARP_CHECK_EQ(frame_idx, m_cur_frame_idx);
			}
		}
		if(is_empty){
			// wait for read image
			usleep(10000);
		}else{
			std::unique_lock<std::mutex> lock(m_mutex);
			rgb_img = m_color_images.front();
			m_color_images.pop_front();
			depth_img = m_depth_images.front();
			m_depth_images.pop_front();
			return;
		}
	}
	LOG(FATAL) << "time out to fetch images" << std::endl;
}

void surfelwarp::VolumeDeformFileFetch::DownSampleImage()
{
	if (d_color_src.empty()) {
		d_color_src = cv::cuda::createContinuous(m_image_rows, m_image_cols, CV_8UC3);
	}
	if (d_depth_src.empty()) {
		d_depth_src = cv::cuda::createContinuous(m_image_rows, m_image_cols, CV_16UC1);
	}

	d_color_src.upload(m_color_img);
	d_depth_src.upload(m_depth_img);

	m_color_img.release();
	m_depth_img.release();

	cv::cuda::pyrDown(d_color_src, d_color_downsampled);
	cv::cuda::pyrDown(d_depth_src, d_depth_downsampled);
	cv::cuda::bilateralFilter(d_depth_downsampled, d_depth_smoothed, 7.0, 20.0, 20.0);

	d_color_downsampled.download(m_color_img);
	d_depth_smoothed.download(m_depth_img);
}

void surfelwarp::VolumeDeformFileFetch::FetchRGBImage(size_t frame_idx, void * rgb_img)
{

}

boost::filesystem::path surfelwarp::VolumeDeformFileFetch::FileNameVolumeDeform(size_t frame_idx, bool is_depth_img) const
{
	//Construct the file_name
	char frame_idx_str[20];
	sprintf(frame_idx_str, "%06d", static_cast<int>(frame_idx));
	std::string file_name = "frame-";
	file_name += std::string(frame_idx_str);
	if (is_depth_img) {
		file_name += ".depth";
	}
	else {
		file_name += ".color";
	}
	file_name += ".png";

	//Construct the path
	path file_path = m_data_path / path(file_name);
	return file_path;
}



boost::filesystem::path surfelwarp::VolumeDeformFileFetch::FileNameSurfelWarp(size_t frame_idx, bool is_depth_img) const {
	return FileNameVolumeDeform(frame_idx, is_depth_img);
}

//std::mutex surfelwarp::VolumeDeformFileFetch::m_mutex;

void surfelwarp::VolumeDeformFileFetch::ReadImageFromFile(){
	while(1) {
		bool is_empty;
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            bool d_empty = m_depth_images.empty();
            bool c_empty = m_color_images.empty();
            if (d_empty != c_empty){
                LOG(FATAL) << "Error: the number of depth_image and color_image is not queal!" << std::endl;
                break;
            }
		    is_empty = d_empty && c_empty;
        }
        if (is_empty){
            std::unique_lock<std::mutex> lock(m_mutex);
			m_cur_frame_idx ++; 

            path depth_file_path = FileNameSurfelWarp(m_cur_frame_idx, true);
            if (! boost::filesystem::exists(depth_file_path)){
				LOG(WARNING) << depth_file_path << " not exists!" << std::endl;
                break;
            }
    	    path rgb_file_path = FileNameSurfelWarp(m_cur_frame_idx, false);
        	if (! boost::filesystem::exists(rgb_file_path)){
				LOG(WARNING) << rgb_file_path << " not exists!" << std::endl;
                break;
            }

        	m_depth_img = cv::imread(depth_file_path.string(), CV_ANYCOLOR | CV_ANYDEPTH);
			m_color_img = cv::imread(rgb_file_path.string(), CV_ANYCOLOR | CV_ANYDEPTH);
			
			// downsample
			if (config.use_downsample()) DownSampleImage();
			
			m_depth_images.push_back(m_depth_img);
			m_color_images.push_back(m_color_img);
			
        } else {
            usleep(3000);
        }
    }
}
