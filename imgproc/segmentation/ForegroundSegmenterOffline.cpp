//
// Created by wei on 5/29/18.
//

#include "common/logging.h"
#include "common/OpenCV_CrossPlatform.h"
#include "common/Constants.h"
#include "common/ConfigParser.h"
#include "common/common_texture_utils.h"
#include "imgproc/segmentation/ForegroundSegmenter.h"
#include "imgproc/segmentation/ForegroundSegmenterOffline.h"


surfelwarp::ForegroundSegmenterOffline::ForegroundSegmenterOffline() {
	//Check the directory
	const auto& config = ConfigParser::Instance();
	m_mask_dir = config.data_path() / "mask";
	SURFELWARP_CHECK(boost::filesystem::exists(m_mask_dir));
	
	//Correct the size of the image
	m_frame_idx = 1;
	m_clip_rows = config.clip_image_rows();
	m_clip_cols = config.clip_image_cols();
}

void surfelwarp::ForegroundSegmenterOffline::AllocateBuffer(unsigned clip_rows, unsigned clip_cols) {
	SURFELWARP_CHECK(m_clip_cols == clip_cols);
	SURFELWARP_CHECK(m_clip_rows == clip_rows);
	
	m_foreground_mask_host = cv::Mat(clip_rows, clip_cols, CV_8UC1);
	createUChar1TextureSurface(clip_rows, clip_cols, m_foreground_mask_collect);
	createFloat1TextureSurface(clip_rows, clip_cols, m_filter_foregound_mask_collect);
}

void surfelwarp::ForegroundSegmenterOffline::ReleaseBuffer() {
	releaseTextureCollect(m_foreground_mask_collect);
	releaseTextureCollect(m_filter_foregound_mask_collect);
}

void surfelwarp::ForegroundSegmenterOffline::SetInputImages(cudaTextureObject_t clip_normalized_rgb_img,
                                                            cudaTextureObject_t raw_depth_img,
                                                            cudaTextureObject_t clip_depth_img, int frame_idx,
                                                            cudaTextureObject_t clip_background_rgb) {
	m_frame_idx = frame_idx;
}


/* The method to upload and filter the offline mask
 */
boost::filesystem::path surfelwarp::ForegroundSegmenterOffline::getSegmentMaskPath(int frame_idx) {
	using path = boost::filesystem::path;
	
	//Construct the file_name
	char frame_idx_str[20];
	sprintf(frame_idx_str, "%06d", frame_idx);
	std::string file_name = "frame-";
	file_name += std::string(frame_idx_str);
	file_name += ".mask.png";
	
	//Construct the path
	path file_path = m_mask_dir / path(file_name);
	return file_path;
}

void surfelwarp::ForegroundSegmenterOffline::Segment(cudaStream_t stream) {
	//Read the image from file
	const auto file_name = getSegmentMaskPath(m_frame_idx);
	m_foreground_mask_host = cv::imread(file_name.string(), CV_ANYCOLOR | CV_ANYDEPTH);
	
	//Upload to texture
	cudaSafeCall(cudaMemcpyToArrayAsync(
		m_foreground_mask_collect.d_array,
		0, 0,
		m_foreground_mask_host.data,
		sizeof(unsigned char) * m_clip_cols * m_clip_rows,
		cudaMemcpyHostToDevice,
		stream
	));
	
	//Perform filtering
	ForegroundSegmenter::FilterForegroundMask(
		m_foreground_mask_collect.texture, 
		m_clip_rows, m_clip_cols, 
		Constants::kForegroundSigma, 
		m_filter_foregound_mask_collect.surface, 
		stream
	);
}



