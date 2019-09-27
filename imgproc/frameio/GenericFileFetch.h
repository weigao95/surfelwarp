//  ================================================================
//  Created by Gregory Kramida on 9/23/19.
//  Copyright (c) 2019 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#pragma once

#include "FetchInterface.h"

#include <boost/filesystem/path.hpp>
#include <vector>
#include <mutex>

namespace surfelwarp
{
	/**
	 * \brief Utility for fetching depth & RGB frames from images on disk, from the directory specified at construction.
	 * \details The image filenames must include 0-based index, there must be the same number of RGB and depth frames,
	 * RGB filenames must include one of {RGB, color, Color, COLOR, rgb} in the filename,
	 * depth filenames must include one of {depth, DEPTH, Depth} in the filename.
	 * If the source directory also contains a matching number of likewise-indexed mask images, with each filename
	 * including one of {mask, Mask, MASK}, the masks are applied to both the RGB and depth images in memory as part
	 * of the fetch operation.
	 */
	class GenericFileFetch : public FetchInterface
	{
	public:
		using Ptr = std::shared_ptr<GenericFileFetch>;
		using path = boost::filesystem::path;

		explicit GenericFileFetch(const path& data_path, std::string file_extension = ".png", bool force_no_masks = false);
		~GenericFileFetch() override = default;

		//Buffer may be maintained outside fetch object for thread safety
		void FetchDepthImage(size_t frame_idx, cv::Mat& depth_img) override;
		void FetchDepthImage(size_t frame_idx, void* depth_img) override;

		//Should be rgb, in CV_8UC3 format
		void FetchRGBImage(size_t frame_idx, cv::Mat& rgb_img) override;
		void FetchRGBImage(size_t frame_idx, void* rgb_img) override;

	private:
		static int GetFrameNumber(const path& filename);
		static bool HasSubstringFromSet(const std::string& string, const std::string* set, int set_size);
		static bool FilenameIndicatesDepthImage(const path& filename, const std::string& valid_extension);
		static bool FilenameIndicatesRGBImage(const path& filename, const std::string& valid_extension);
		static bool FilenameIndicatesMaskImage(const path& filename, const std::string& valid_extension);

		std::vector<path> m_rgb_image_paths;
		std::vector<path> m_depth_image_paths;
		std::vector<path> m_mask_image_paths;

		size_t m_mask_buffer_ix;
		cv::Mat m_mask_image_buffer;
		bool m_use_masks;
		std::mutex mask_mutex;
	};
} // end namespace surfelwarp

