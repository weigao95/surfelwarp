#pragma once
#include "imgproc/FetchInterface.h"
#include <string>
#include <boost/filesystem.hpp>

namespace surfelwarp
{
	class FileFetch : public FetchInterface
	{
	public:
		using Ptr = std::shared_ptr<FileFetch>;
		using path = boost::filesystem::path;

		//Just copy the string to data path
		explicit FileFetch(
			const path& data_path
		) : m_data_path(data_path) {}

		
		~FileFetch() = default;

		//Main interface
		void FetchDepthImage(size_t frame_idx, cv::Mat& depth_img) override;
		void FetchDepthImage(size_t frame_idx, void* depth_img) override;
		void FetchRGBImage(size_t frame_idx, cv::Mat& rgb_img) override;
		void FetchRGBImage(size_t frame_idx, void* rgb_img) override;

	private:
		path m_data_path; //The path prefix for the data

		//A series of naming functions
		path FileNameVolumeDeform(size_t frame_idx, bool is_depth_img) const;
		path FileNameSurfelWarp(size_t frame_idx, bool is_depth_img) const;
	};



}