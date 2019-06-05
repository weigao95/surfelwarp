#pragma once
#include <opencv2/opencv.hpp>
#include <memory>

namespace surfelwarp {
	
	/**
	 * \brief The virtual class for all the input image fetching.
	 *        The implementation should support threaded fetching.
	 */
	class FetchInterface
	{
	public:
		using Ptr = std::shared_ptr<FetchInterface>;

		//Default contruct and de-construct
		FetchInterface() = default;
		virtual ~FetchInterface() = default;

		//Do not allow copy/assign/move
		FetchInterface(const FetchInterface&) = delete;
		FetchInterface(FetchInterface&&) = delete;
		FetchInterface& operator=(const FetchInterface&) = delete;
		FetchInterface& operator=(FetchInterface&&) = delete;

		//Buffer may be maintained outside fetch object for thread safety
		virtual void FetchDepthImage(size_t frame_idx, cv::Mat& depth_img) = 0;
		virtual void FetchDepthImage(size_t frame_idx, void* depth_img) = 0;

		//Should be rgb, in CV_8UC3 format
		virtual void FetchRGBImage(size_t frame_idx, cv::Mat& rgb_img) = 0;
		virtual void FetchRGBImage(size_t frame_idx, void* rgb_img) = 0;
	};

}