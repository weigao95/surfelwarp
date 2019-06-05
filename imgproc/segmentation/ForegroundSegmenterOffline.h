//
// Created by wei on 5/29/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "imgproc/segmentation/ForegroundSegmenter.h"

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

namespace surfelwarp {
	
	/**
	 * \brief The offline segmenter just load the
	 *        foreground mask from disk, perform suitable
	 *        upsampling and filtering
	 */
	class ForegroundSegmenterOffline : public ForegroundSegmenter {
	private:
		//The filename for segment mask
		boost::filesystem::path m_mask_dir;
		boost::filesystem::path getSegmentMaskPath(int frame_idx);
		
		//The only used input is frame idx
		int m_frame_idx;
		unsigned m_clip_rows, m_clip_cols;
		
	public:
		//Update the pointer
		using Ptr = std::shared_ptr<ForegroundSegmenterOffline>;
		ForegroundSegmenterOffline();
		~ForegroundSegmenterOffline() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(ForegroundSegmenterOffline);
		
		void AllocateBuffer(unsigned clip_rows, unsigned clip_cols) override;
		void ReleaseBuffer() override;
		
		void SetInputImages(
			cudaTextureObject_t clip_normalized_rgb_img,
			cudaTextureObject_t raw_depth_img,
			cudaTextureObject_t clip_depth_img,
			int frame_idx,
			cudaTextureObject_t clip_background_rgb = 0
		) override;
	
		
	private:
		cv::Mat m_foreground_mask_host;
		CudaTextureSurface m_foreground_mask_collect; //uchar texture, original(clip) resolution
		CudaTextureSurface m_filter_foregound_mask_collect; //float1 texture
	
	public:
		void Segment(cudaStream_t stream = 0) override;
		cudaTextureObject_t ForegroundMask() const override { return m_foreground_mask_collect.texture; }
		cudaTextureObject_t FilterForegroundMask() const override { return m_filter_foregound_mask_collect.texture; }
	};
}