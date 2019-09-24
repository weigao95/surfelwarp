#pragma once
#include "FetchInterface.h"
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
		explicit VolumeDeformFileFetch(
			const path& data_path
		) : m_data_path(data_path) {}


		~VolumeDeformFileFetch() = default;

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