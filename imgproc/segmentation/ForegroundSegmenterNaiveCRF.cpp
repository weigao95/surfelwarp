//
// Created by wei on 2/24/18.
//

#include "common/Constants.h"
#include "imgproc/segmentation/ForegroundSegmenterNaiveCRF.h"
#include "imgproc/segmentation/foreground_crf_window.h"
#include "imgproc/segmentation/crf_common.h"
#include "common/common_texture_utils.h"
#include "visualization/Visualizer.h"

void surfelwarp::ForegroundSegmenterNaiveCRF::AllocateBuffer(
	unsigned clip_rows,
	unsigned clip_cols
) {
	//Do subsampling here
	const auto subsampled_rows = clip_rows / crf_subsample_rate;
	const auto subsampled_cols = clip_cols / crf_subsample_rate;
	
	//Allocate the buffer for unary energy
	m_unary_energy_map_subsampled.create(subsampled_rows, subsampled_cols);

	//Allocate the buffer for meanfield q
	createFloat1TextureSurface(subsampled_rows, subsampled_cols, m_meanfield_foreground_collect_subsampled[0]);
	createFloat1TextureSurface(subsampled_rows, subsampled_cols, m_meanfield_foreground_collect_subsampled[1]);

	//Allocate the buffer for segmentation mask
	createUChar1TextureSurface(subsampled_rows, subsampled_cols, m_segmented_mask_collect_subsampled);
	
	//Allocate the upsampled buffer
	createUChar1TextureSurface(clip_rows, clip_cols, m_foreground_mask_collect_upsampled);
	createFloat1TextureSurface(clip_rows, clip_cols, m_filter_foreground_mask_collect_upsampled);
}

void surfelwarp::ForegroundSegmenterNaiveCRF::ReleaseBuffer()
{
	m_unary_energy_map_subsampled.release();
	releaseTextureCollect(m_meanfield_foreground_collect_subsampled[0]);
	releaseTextureCollect(m_meanfield_foreground_collect_subsampled[1]);
	releaseTextureCollect(m_segmented_mask_collect_subsampled);
}

void surfelwarp::ForegroundSegmenterNaiveCRF::SetInputImages(
	cudaTextureObject_t clip_normalized_rgb_img,
	cudaTextureObject_t raw_depth_img,
	cudaTextureObject_t clip_depth_img,
	int frame_idx,
	cudaTextureObject_t clip_background_rgb
) {
	m_input_texture.clip_normalize_rgb_img = clip_normalized_rgb_img;
	m_input_texture.raw_depth_img = raw_depth_img;
	m_input_texture.clip_depth_img = clip_depth_img;
}

void surfelwarp::ForegroundSegmenterNaiveCRF::Segment(cudaStream_t stream)
{
	//First init the unary energy and meanfield
	initMeanfieldUnaryEnergy(stream);
	
	//The main loop
	const auto max_iters = Constants::kMeanfieldSegmentIteration;
	for(auto i = 0; i < max_iters; i++) {
		//saveMeanfieldApproximationMap(i);
		inferenceIteration(stream);
	}
	
	//Write the output to segmentation mask
	writeSegmentationMask(stream);
	upsampleFilterForegroundMask(stream);
}

void surfelwarp::ForegroundSegmenterNaiveCRF::initMeanfieldUnaryEnergy(cudaStream_t stream) 
{
	initMeanfieldUnaryForegroundSegmentation(
		m_input_texture.raw_depth_img,
		m_input_texture.clip_depth_img,
		m_unary_energy_map_subsampled,
		m_meanfield_foreground_collect_subsampled[0].surface,
		stream
	);
	m_updated_meanfield_idx = 0;
}


void surfelwarp::ForegroundSegmenterNaiveCRF::inferenceIteration(cudaStream_t stream) {
	//The index value
	const auto input_idx = m_updated_meanfield_idx % 2;
	const int output_idx = (input_idx + 1) % 2;
	m_updated_meanfield_idx = (m_updated_meanfield_idx + 1) % 2;

	//the constant value for apperance kernel
	const float apperance_weight = 0.5f;
	const float sigma_alpha = 10;
	const float sigma_beta = 15;

	//The constant value for smooth kernel
	const float sigma_gamma = 3;
	const float smooth_weight = 0.5f;
	foregroundMeanfieldIterWindow(
		m_meanfield_foreground_collect_subsampled[input_idx].texture,
		m_input_texture.clip_normalize_rgb_img,
		m_unary_energy_map_subsampled,
		sigma_alpha, 
		sigma_beta,
		sigma_gamma,
		apperance_weight,
		smooth_weight,
		m_meanfield_foreground_collect_subsampled[output_idx].surface,
		stream
	);
}

void surfelwarp::ForegroundSegmenterNaiveCRF::writeSegmentationMask(cudaStream_t stream) {
	const auto write_idx = m_updated_meanfield_idx % 2;
	writeForegroundSegmentationMask(
		m_meanfield_foreground_collect_subsampled[write_idx].texture,
		m_unary_energy_map_subsampled.rows(), m_unary_energy_map_subsampled.cols(),
		m_segmented_mask_collect_subsampled.surface,
		stream
	);
}

void surfelwarp::ForegroundSegmenterNaiveCRF::upsampleFilterForegroundMask(cudaStream_t stream) {
	ForegroundSegmenter::UpsampleFilterForegroundMask(
		m_segmented_mask_collect_subsampled.texture,
		m_unary_energy_map_subsampled.rows(), m_unary_energy_map_subsampled.cols(),
		crf_subsample_rate,
		Constants::kForegroundSigma,
		m_foreground_mask_collect_upsampled.surface,
		m_filter_foreground_mask_collect_upsampled.surface,
		stream
	);
}


void surfelwarp::ForegroundSegmenterNaiveCRF::saveMeanfieldApproximationMap(const unsigned iter)
{
	std::stringstream ss;
	ss << iter;
	std::string file_name = "meanfield-";
	file_name += ss.str();
	file_name += ".png";
	Visualizer::SaveBinaryMeanfield(m_meanfield_foreground_collect_subsampled[m_updated_meanfield_idx].texture, file_name);
}