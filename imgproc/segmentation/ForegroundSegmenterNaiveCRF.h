//
// Created by wei on 2/24/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "imgproc/segmentation/ForegroundSegmenter.h"


namespace surfelwarp {


	class ForegroundSegmenterNaiveCRF : public ForegroundSegmenter {
	private:
		//The buffer for mean field probability, float1 texture
		CudaTextureSurface m_meanfield_foreground_collect_subsampled[2];
		int m_updated_meanfield_idx;
		
		//The output as a binary mask
		CudaTextureSurface m_segmented_mask_collect_subsampled;

		//The buffer for unary energy
		DeviceArray2D<float2> m_unary_energy_map_subsampled;

		//Small struct to collect the input
		struct InputTexture {
			cudaTextureObject_t clip_normalize_rgb_img;
			cudaTextureObject_t raw_depth_img;
			cudaTextureObject_t clip_depth_img;
		} m_input_texture;

	public:
		//Update the pointer type
		using Ptr = std::shared_ptr<ForegroundSegmenterNaiveCRF>;
		
		//Constructor and destrctor DO NOT allocate and release the buffer
		SURFELWARP_DEFAULT_CONSTRUCT_DESTRUCT(ForegroundSegmenterNaiveCRF);
		SURFELWARP_NO_COPY_ASSIGN_MOVE(ForegroundSegmenterNaiveCRF);

		//Buffer management
		void AllocateBuffer(unsigned clip_rows, unsigned clip_cols) override;
		void ReleaseBuffer() override;

		//The main interface
		void SetInputImages(
			cudaTextureObject_t clip_normalized_rgb_img, 
			cudaTextureObject_t raw_depth_img, 
			cudaTextureObject_t clip_depth_img,
			int frame_idx,
			cudaTextureObject_t clip_background_rgb = 0
		) override;
		void Segment(cudaStream_t stream = 0) override;
		cudaTextureObject_t MeanFieldMap() const { return m_meanfield_foreground_collect_subsampled[m_updated_meanfield_idx].texture; }
		
		//Detailed implementation
		void initMeanfieldUnaryEnergy(cudaStream_t stream = 0);
		void inferenceIteration(cudaStream_t stream = 0);
		void writeSegmentationMask(cudaStream_t stream = 0);
		
		
		//Perform upsampling and filtering of the mask
	private:
		CudaTextureSurface m_foreground_mask_collect_upsampled;
		CudaTextureSurface m_filter_foreground_mask_collect_upsampled;
	public:
		void upsampleFilterForegroundMask(cudaStream_t stream = 0);
		cudaTextureObject_t ForegroundMask() const override { return m_foreground_mask_collect_upsampled.texture; }
		cudaTextureObject_t FilterForegroundMask() const override { return m_filter_foreground_mask_collect_upsampled.texture; }
		

		//Debug method
		void saveMeanfieldApproximationMap(const unsigned iter = 0);
	};

}

