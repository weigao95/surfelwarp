// #pragma once
// #include "FetchInterface.h"
// #include <string>
// #include <boost/filesystem.hpp>
// #include <k4a/k4a.hpp>

// namespace surfelwarp
// {
// 	/**
// 	 * \brief Utility for fetching depth & RGB frames from azure kinect dk camera
// 	 */
// 	class AzureKinectDKFetch : public FetchInterface
// 	{
// 	public:
// 		using Ptr = std::shared_ptr<AzureKinectDKFetch>;
// 		using path = boost::filesystem::path;

// 		// todo: load device config from file
// 		explicit AzureKinectDKFetch(const path& data_path, bool save_online_frame);


// 		~AzureKinectDKFetch();

// 		//Main interface
// 		void FetchDepthImage(size_t frame_idx, cv::Mat& depth_img) override;
// 		void FetchDepthImage(size_t frame_idx, void* depth_img) override;
// 		void FetchRGBImage(size_t frame_idx, cv::Mat& rgb_img) override;
// 		void FetchRGBImage(size_t frame_idx, void* rgb_img) override;

// 		void FetchDepthAndRGBImage(size_t frame_idx, cv::Mat& depth_img, cv::Mat& rgb_img) override;

// 	private:
// 		// kinect dk handle
// 		k4a::device m_device;
//         k4a::capture m_capture;
// 		k4a::transformation m_transformation;
// 		k4a::image m_k4a_depth_image;
// 		k4a::image m_k4a_color_image;
// 		k4a::image m_k4a_transformed_depth_image;

// 		// images 
// 		cv::Mat m_depth_image;
// 		cv::Mat m_color_image_rgba;
// 		cv::Mat m_color_image_rgb;
// 		cv::Mat m_color_image;
// 		cv::Mat m_depth_image_undistorted;
// 		cv::Mat m_color_image_undistorted;

// 		// undistort
// 		cv::Mat m_camera_matrix;
// 		cv::Mat m_new_camera_matrix;
// 		cv::Mat m_map1;
// 		cv::Mat m_map2;


// 		std::vector<cv::Mat> m_depth_image_vec;
// 		std::vector<cv::Mat> m_color_image_vec;



// 		// frame property
// 		size_t m_cur_frame_num;
// 		size_t m_frame_height_pixels;
// 		size_t m_frame_width_pixels;
// 		cv::Size m_frame_size;

// 		// save the frame 
// 		bool m_save_online_frame;
        
//         // scale factor
//         float m_scale_factor;

// 		path m_data_path;


//         bool TakeNewPictureFrame();

// 		void DownscaleCalibration(const k4a::calibration calibration, k4a::calibration &new_calibration, float scale = 2.0);

// 	};
// }