#pragma once
#include "FetchInterface.h"
#include <mutex>
#include <thread>
#include <string>
#include <boost/filesystem.hpp>

namespace surfelwarp
{
	/**
	 * \brief Utility for fetching depth & RGB frames specifically in the format of the VolumeDeform dataset, i.e.
	 * depth frames named as "frame-000000.depth.png" and RBG frames named as "frame-000000.color.png", where zeros are
	 * replaced by the zero-based frame index padded on the left by zeroes to be 6 characters long.
	 */
	class VolumeDeformFileFetch : public FetchInterface
	{
	public:
		using Ptr = std::shared_ptr<VolumeDeformFileFetch>;
		using path = boost::filesystem::path;

		//Just copy the string to data path
		explicit VolumeDeformFileFetch(const path& data_path);


		~VolumeDeformFileFetch() = default;

		//Main interface
		void FetchDepthImage(size_t frame_idx, cv::Mat& depth_img) override;
		void FetchDepthImage(size_t frame_idx, void* depth_img) override;
		void FetchRGBImage(size_t frame_idx, cv::Mat& rgb_img) override;
		void FetchRGBImage(size_t frame_idx, void* rgb_img) override;
		void FetchDepthAndRGBImage(size_t frame_idx, cv::Mat& depth_img, cv::Mat& rgb_img) override;
		void DownSampleImage() override;

	private:
		path m_data_path; //The path prefix for the data

		//A series of naming functions
		path FileNameVolumeDeform(size_t frame_idx, bool is_depth_img) const;
		path FileNameSurfelWarp(size_t frame_idx, bool is_depth_img) const;

		size_t m_cur_frame_idx;
		std::thread m_thread;
		std::list<cv::Mat> m_depth_images;
		std::list<cv::Mat> m_color_images;
		cv::Mat m_depth_img;
		cv::Mat m_color_img;
		cv::cuda::GpuMat d_color_src;
		cv::cuda::GpuMat d_color_downsampled;
		cv::cuda::GpuMat d_depth_smoothed;
		cv::cuda::GpuMat d_depth_src;
		cv::cuda::GpuMat d_depth_downsampled;
		int m_image_rows;
		int m_image_cols;
		
		void ReadImageFromFile();
		std::mutex m_mutex;
		ConfigParser& config = ConfigParser::Instance();
	};
}
