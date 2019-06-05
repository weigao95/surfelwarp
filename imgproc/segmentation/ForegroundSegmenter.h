//
// Created by wei on 2/24/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include <memory>

namespace surfelwarp {
	
	/**
	 * \brief A virtual class for the interface
	 *        to segment the foreground of an image
	 */
	class ForegroundSegmenter {
	public:
		//Reference as pointer
		using Ptr = std::shared_ptr<ForegroundSegmenter>;
		
		//Default constructor, no copy/move/assign
		explicit ForegroundSegmenter() = default;
		virtual ~ForegroundSegmenter() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(ForegroundSegmenter);
		
		//Explicit memory allcoate and release. The caller
		//ensure 1-to-1 allocate-release
		virtual void AllocateBuffer(unsigned cilp_rows, unsigned clip_cols) = 0;
		virtual void ReleaseBuffer() = 0;
		
		//The input-output method
		virtual void SetInputImages(
			cudaTextureObject_t clip_normalized_rgb_img, 
			cudaTextureObject_t raw_depth_img, 
			cudaTextureObject_t clip_depth_img,
			int frame_idx,
			cudaTextureObject_t clip_background_rgb = 0
		) = 0;
		virtual void Segment(cudaStream_t stream = 0) = 0;
		virtual cudaTextureObject_t ForegroundMask() const = 0;
		virtual cudaTextureObject_t FilterForegroundMask() const = 0;
		
		//The method for upsampling and filter the foreground mask
		static void UpsampleFilterForegroundMask(
			cudaTextureObject_t subsampled_mask,
			unsigned subsampled_rows, unsigned subsampled_cols,
			unsigned subsample_rate,
			float sigma,
			cudaSurfaceObject_t upsampled_mask, //uchar texture
			cudaSurfaceObject_t filter_mask, //float1 texture
			cudaStream_t stream = 0
		);

		//The method for filtering
		static void FilterForegroundMask(
			cudaTextureObject_t foreground_mask,
			unsigned mask_rows, unsigned mask_cols,
			float sigma,
			cudaSurfaceObject_t filter_mask,
			cudaStream_t stream = 0
		);
	};
}