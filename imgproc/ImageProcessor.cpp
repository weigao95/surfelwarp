#include "common/common_types.h"
#include "common/common_utils.h"
#include "common/Constants.h"
#include "common/ConfigParser.h"
#include "common/common_texture_utils.h"
#include "imgproc/ImageProcessor.h"
#include "imgproc/depth_clip_filter.h"
#include "imgproc/rgb_clip_normalize.h"
#include "imgproc/depth_surfel_collect.h"
#include "imgproc/generate_maps.h"
#include "imgproc/image_gradient.h"

//The header for segmentation
#include "imgproc/segmentation/ForegroundSegmenterNaiveCRF.h"
#include "imgproc/segmentation/ForegroundSegmenterPermutohedral.h"
#include "imgproc/segmentation/ForegroundSegmenterOffline.h"

//The header for correspondence
#include "imgproc/correspondence/PatchColliderRGBCorrespondence.h"

#include <memory>


surfelwarp::ImageProcessor::ImageProcessor(const surfelwarp::FetchInterface::Ptr &fetcher) : m_image_fetcher(fetcher) {
	//Access the singleton
	const auto& config = ConfigParser::Instance();
	
	//Set the global constants
	m_raw_img_rows = config.raw_image_rows();
	m_raw_img_cols = config.raw_image_cols();
	m_clip_img_rows = config.clip_image_rows();
	m_clip_img_cols = config.clip_image_cols();
	m_clip_near = config.clip_near_mm();
	m_clip_far = config.clip_far_mm();
	
	//The intrinsic parameters
	m_raw_depth_intrinsic = config.depth_intrinsic_raw();
	m_raw_rgb_intrinsic = config.rgb_intrinsic_raw();
	m_clip_rgb_intrinsic = config.rgb_intrinsic_clip();
	m_depth2rgb = config.depth2rgb_dev();
	
	//Invoke the sub-init functions
	allocateFetchBuffer();
	allocateDepthTexture();
	allocateRGBBuffer();
	allocateGeometryTexture();
	allocateColorTimeTexture();
	allocateValidSurfelSelectionBuffer();
	allocateForegroundSegmentationBuffer();
	allocateFeatureCorrespondenceBuffer();
	allocateGradientMap();

	//Init the stream
	initProcessorStream();
}

surfelwarp::ImageProcessor::~ImageProcessor()
{
	releaseFetchBuffer();
	releaseDepthTexture();
	releaseRGBBuffer();
	releaseGeometryTexture();
	releaseColorTimeTexture();
	releaseValidSurfelSelectionBuffer();
	releaseForegroundSegmentationBuffer();
	releaseFeatureCorrespondenceBuffer();
	releaseGradientMap();

	//Release the stream
	releaseProcessorStream();
}

/* Interface functions
 */
surfelwarp::DeviceArrayView<surfelwarp::DepthSurfel> surfelwarp::ImageProcessor::ProcessFirstFrameSerial(
	size_t frame_idx,
	cudaStream_t stream
) {
	FetchFrame(frame_idx);
	UploadDepthImage(stream);
	UploadRawRGBImage(stream);
	
	//This seems cause some problem ,disable it at first
	//ReprojectDepthToRGB(stream);
	ClipFilterDepthImage(stream);
	ClipNormalizeRGBImage(stream);
	
	//The geometry map
	BuildVertexConfigMap(stream);
	BuildNormalRadiusMap(stream);
	BuildColorTimeTexture(frame_idx, stream);
	
	//Collect it
	CollectValidDepthSurfel(stream);
	
	//Ready to return it
	const auto& surfel_array = ValidDepthSurfelArray();
	return DeviceArrayView<DepthSurfel>(surfel_array.ptr(), surfel_array.size());
}



void surfelwarp::ImageProcessor::ProcessFrameSerial(
	CameraObservation & observation, 
	size_t frame_idx, 
	cudaStream_t stream
) {
	FetchFrame(frame_idx);
	UploadDepthImage(stream);
	UploadRawRGBImage(stream);

	//This seems cause some problem ,disable it at first
	//ReprojectDepthToRGB(stream);
	
	ClipFilterDepthImage(stream);
	ClipNormalizeRGBImage(stream);

	//The geometry map
	BuildVertexConfigMap(stream);
	BuildNormalRadiusMap(stream);
	BuildColorTimeTexture(frame_idx, stream);

	//Further computations
	SegmentForeground(frame_idx, stream);
	FindCorrespondence(stream);
	ComputeGradientMap(stream);

	//Write to result
	memset(&observation, 0, sizeof(observation));

	//The raw depth image for visualization
	observation.raw_depth_img = RawDepthTexture();
	
	//The geometry maps
	observation.filter_depth_img = FilteredDepthTexture();
	observation.vertex_config_map = VertexConfidTexture();
	observation.normal_radius_map = NormalRadiusTexture();

	//The color maps
	observation.color_time_map = ColorTimeTexture();
	observation.normalized_rgba_map = ClipNormalizedRGBTexture();
	observation.normalized_rgba_prevframe = ClipNormalizedRGBTexturePrev();
	observation.density_map = DensityMapTexture();
	//observation.density_map_prevframe = DensityMapTexturePrev();
	observation.density_gradient_map = DensityGradientTexture();
	
	//The foreground masks
	observation.foreground_mask = ForegroundMask();
	observation.filter_foreground_mask = FilterForegroundMask();
	observation.foreground_mask_gradient_map = ForegroundMaskGradientTexture();

	//The correspondence pixel pairs
	const auto& pixel_pair_array = CorrespondencePixelPair();
	observation.correspondence_pixel_pairs = DeviceArrayView<ushort4>(pixel_pair_array.ptr(), pixel_pair_array.size());
}


/**
 * The image fetching methods implementation.
 */
void surfelwarp::ImageProcessor::allocateFetchBuffer()
{
	//First allocate the buffer
	const auto raw_img_size = m_raw_img_rows * m_raw_img_cols;
	cudaSafeCall(cudaMallocHost(&m_depth_buffer_pagelock, sizeof(unsigned short) * raw_img_size));
	cudaSafeCall(cudaMallocHost(&m_rgb_buffer_pagelock, sizeof(uchar4) * raw_img_size));
	cudaSafeCall(cudaMallocHost(&m_rgb_prev_buffer_pagelock, sizeof(uchar4) * raw_img_size));
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());

	//Construct the opencv matrix
	m_depth_img = cv::Mat(cv::Size(m_raw_img_cols, m_raw_img_rows), CV_16UC1);
	m_rgb_img = cv::Mat(cv::Size(m_raw_img_cols, m_raw_img_rows), CV_8UC4);
	m_rgb_img_prev = cv::Mat(cv::Size(m_raw_img_cols, m_raw_img_rows), CV_8UC4);
}

void surfelwarp::ImageProcessor::releaseFetchBuffer()
{
	cudaSafeCall(cudaFreeHost(m_depth_buffer_pagelock));
	cudaSafeCall(cudaFreeHost(m_rgb_buffer_pagelock));
	cudaSafeCall(cudaFreeHost(m_rgb_prev_buffer_pagelock));
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
}

void surfelwarp::ImageProcessor::FetchFrame(size_t frame_idx)
{
	FetchDepthImage(frame_idx);
	FetchRGBImage(frame_idx);
	FetchRGBPrevFrame(frame_idx);
}

void surfelwarp::ImageProcessor::FetchDepthImage(size_t frame_idx)
{
	m_image_fetcher->FetchDepthImage(frame_idx, m_depth_img);
	//Must explict perform this copy?
	memcpy(m_depth_buffer_pagelock, m_depth_img.data, 
		sizeof(unsigned short) * m_raw_img_cols * m_raw_img_rows
	);
}

void surfelwarp::ImageProcessor::FetchRGBImage(size_t frame_idx)
{
	m_image_fetcher->FetchRGBImage(frame_idx, m_rgb_img);
	memcpy(m_rgb_buffer_pagelock, m_rgb_img.data,
		sizeof(uchar3) * m_raw_img_rows * m_raw_img_cols
	);
}

void surfelwarp::ImageProcessor::FetchRGBPrevFrame(size_t curr_frame_idx)
{
	//First compute the frame idx
	size_t prev_frame_idx;
	if (curr_frame_idx == 0) prev_frame_idx = curr_frame_idx;
	else prev_frame_idx = curr_frame_idx - 1;

	//Fetch it
	m_image_fetcher->FetchRGBImage(prev_frame_idx, m_rgb_img_prev);
	memcpy(m_rgb_prev_buffer_pagelock, m_rgb_img_prev.data,
		sizeof(uchar3) * m_raw_img_rows * m_raw_img_cols
	);
}

/**
* The depth texture and filtering methods implementation.
*/
void surfelwarp::ImageProcessor::allocateDepthTexture()
{
	//The raw image shoud use raw_size
	createDepthTextureSurface(
		m_raw_img_rows, m_raw_img_cols,
		m_depth_raw_collect
	);

	//The filtered image should use clip_size
	createDepthTextureSurface(
		m_clip_img_rows, m_clip_img_cols,
		m_depth_filter_collect
	);

	//Create the reprojected depth buffer
	m_reprojected_buffer.create(
		Constants::kReprojectScaleFactor * m_raw_img_rows,
		Constants::kReprojectScaleFactor * m_raw_img_cols
	);
}

void surfelwarp::ImageProcessor::releaseDepthTexture()
{
	//Release the reprojection buffer
	m_reprojected_buffer.release();

    //Next release the cuda array
	releaseTextureCollect(m_depth_raw_collect);
	releaseTextureCollect(m_depth_filter_collect);
}

void surfelwarp::ImageProcessor::UploadDepthImage(cudaStream_t stream)
{
	cudaSafeCall(cudaMemcpyToArrayAsync(
		m_depth_raw_collect.d_array,
		0, 0,
		m_depth_buffer_pagelock,
		sizeof(unsigned short) * m_raw_img_cols * m_raw_img_rows,
		cudaMemcpyHostToDevice,
		stream
	));
}

void surfelwarp::ImageProcessor::ReprojectDepthToRGB(cudaStream_t stream)
{
	//First clear to reprojection buffer
	cudaSafeCall(cudaMemset2DAsync(
		m_reprojected_buffer.ptr(),
		m_reprojected_buffer.colsBytes(),
		0,
		m_reprojected_buffer.cols(),
		m_reprojected_buffer.rows(),
		stream
	));
	
	//Do reprojection
	const auto depth_raw_intrinsic_inv = inverse(m_raw_depth_intrinsic);
	reprojectDepthToRGB(
		m_depth_raw_collect.texture, m_depth_raw_collect.surface,
		m_reprojected_buffer,
		m_raw_img_rows, m_raw_img_cols,
		depth_raw_intrinsic_inv, m_raw_rgb_intrinsic, 
		m_depth2rgb, 
		stream
	);
}

void surfelwarp::ImageProcessor::ClipFilterDepthImage(cudaStream_t stream) {
	clipFilterDepthImage(
		m_depth_raw_collect.texture, 
		m_clip_img_rows, m_clip_img_cols, 
		m_clip_near, m_clip_far,
		m_depth_filter_collect.surface, 
		stream
	);
}


/**
* The RGB cliping and normalization methods implementation.
*/
void surfelwarp::ImageProcessor::allocateRGBBuffer()
{
    //The raw buffer for input buffer
	m_raw_rgb_buffer.create(m_raw_img_rows * m_raw_img_cols);
	m_raw_rbg_buffer_prev.create(m_raw_img_rows * m_raw_img_cols);

    //The texture and surface of cliped & normalized rgb image
    createFloat4TextureSurface(
		m_clip_img_rows, m_clip_img_cols,
        m_clip_normalize_rgb_collect
    );
	createFloat4TextureSurface(
		m_clip_img_rows, m_clip_img_cols, 
		m_clip_normalize_rgb_collect_prev
	);

	//The texture and surface for denstiy map, in the same size as
	//clip_normalized rgb image
	createFloat1TextureSurface(m_clip_img_rows, m_clip_img_cols, m_density_map_collect);
	createFloat1TextureSurface(m_clip_img_rows, m_clip_img_cols, m_filter_density_map_collect);
}

void surfelwarp::ImageProcessor::releaseRGBBuffer()
{
    //Release the buffer
	m_raw_rgb_buffer.release();

	//Release the texture
	releaseTextureCollect(m_clip_normalize_rgb_collect_prev);
	releaseTextureCollect(m_clip_normalize_rgb_collect);
	releaseTextureCollect(m_density_map_collect);
	releaseTextureCollect(m_filter_density_map_collect);
}

void surfelwarp::ImageProcessor::UploadRawRGBImage(cudaStream_t stream)
{
	void* ptr = m_raw_rgb_buffer.ptr();
	//This should be aligned due to the size
	cudaSafeCall(cudaMemcpyAsync(
		ptr, m_rgb_buffer_pagelock,
		sizeof(uchar3) * m_raw_img_cols * m_raw_img_rows, 
		cudaMemcpyHostToDevice,
		stream
	));

	ptr = m_raw_rbg_buffer_prev.ptr();
	//This should be aligned due to the size
	cudaSafeCall(cudaMemcpyAsync(
		ptr, m_rgb_prev_buffer_pagelock,
		sizeof(uchar3) * m_raw_img_cols * m_raw_img_rows,
		cudaMemcpyHostToDevice,
		stream
	));
}

void surfelwarp::ImageProcessor::ClipNormalizeRGBImage(cudaStream_t stream)
{
	clipNormalizeRGBImage(
		m_raw_rgb_buffer, 
		m_clip_img_rows, m_clip_img_cols, 
		m_clip_normalize_rgb_collect.surface,
		m_density_map_collect.surface,
		stream
	);
	
	filterDensityMap(
		m_density_map_collect.texture,
		m_filter_density_map_collect.surface,
		m_clip_img_rows, m_clip_img_cols,
		stream
	);

	clipNormalizeRGBImage(
		m_raw_rbg_buffer_prev, 
		m_clip_img_rows, m_clip_img_cols,
		m_clip_normalize_rgb_collect_prev.surface, 
		//m_density_map_collect_prev.surface,
		stream
	);
}


/**
* The methods for building geometry (vertex, normal and radius) maps
*/
void surfelwarp::ImageProcessor::allocateGeometryTexture()
{
	//The texture and surface of cliped & normalized rgb image
	createFloat4TextureSurface(
		m_clip_img_rows, m_clip_img_cols,
		m_vertex_confid_collect
	);
	createFloat4TextureSurface(
		m_clip_img_rows, m_clip_img_cols,
		m_normal_radius_collect
	);
}

void surfelwarp::ImageProcessor::releaseGeometryTexture()
{
	releaseTextureCollect(m_vertex_confid_collect);
	releaseTextureCollect(m_normal_radius_collect);
}


void surfelwarp::ImageProcessor::BuildVertexConfigMap(cudaStream_t stream)
{
	const IntrinsicInverse clip_intrinsic_inv = inverse(m_clip_rgb_intrinsic);
	createVertexConfigMap(
		m_depth_filter_collect.texture,
		m_clip_img_rows, m_clip_img_cols,
		clip_intrinsic_inv,
		m_vertex_confid_collect.surface, 
		stream
	);
}

void surfelwarp::ImageProcessor::BuildNormalRadiusMap(cudaStream_t stream) {
	createNormalRadiusMap(
		m_vertex_confid_collect.texture, 
		m_clip_img_rows, m_clip_img_cols,
		m_normal_radius_collect.surface,
		stream
	);
}


/**
* The methods for building color_time maps
*/
void surfelwarp::ImageProcessor::allocateColorTimeTexture()
{
	createFloat4TextureSurface(m_clip_img_rows, m_clip_img_cols, m_color_time_collect);
}

void surfelwarp::ImageProcessor::releaseColorTimeTexture()
{
	releaseTextureCollect(m_color_time_collect);
}

void surfelwarp::ImageProcessor::BuildColorTimeTexture(size_t frame_idx ,cudaStream_t stream)
{
	const float init_time = float(frame_idx);
	createColorTimeMap(
		m_raw_rgb_buffer, 
		m_clip_img_rows, m_clip_img_cols, 
		init_time, 
		m_color_time_collect.surface, 
		stream
	);
}

/**
* The methods for select and collect valid depth surfel
*/
void surfelwarp::ImageProcessor::allocateValidSurfelSelectionBuffer()
{
	const auto num_pixels = m_clip_img_cols * m_clip_img_rows;
	m_valid_depth_pixel_selector.AllocateAndInit(num_pixels);
	m_depth_surfel.AllocateBuffer(num_pixels);
}

void surfelwarp::ImageProcessor::releaseValidSurfelSelectionBuffer()
{

}

void surfelwarp::ImageProcessor::CollectValidDepthSurfel(cudaStream_t stream)
{
	//Construct the array
	const auto num_pixels = m_clip_img_cols * m_clip_img_rows;
	DeviceArray<char> valid_indicator = DeviceArray<char>(
		m_valid_depth_pixel_selector.select_indicator_buffer.ptr(),
		num_pixels
	);
	
	//First mark the validity of depth surfels
	markValidDepthPixel(
		m_depth_filter_collect.texture,
		m_clip_img_rows, m_clip_img_cols,
		m_valid_depth_pixel_selector.select_indicator_buffer,
		stream
	);
	
	//Do selection
	m_valid_depth_pixel_selector.Select(valid_indicator, stream);
	
	//Construct the output
	const auto selected_surfel_size = m_valid_depth_pixel_selector.valid_selected_idx.size();
	m_depth_surfel.ResizeArrayOrException(selected_surfel_size);
	
	//Collect it
	DeviceArray<DepthSurfel> valid_surfel_array = m_depth_surfel.Array();
	collectDepthSurfel(
		m_vertex_confid_collect.texture,
		m_normal_radius_collect.texture,
		m_color_time_collect.texture,
		m_valid_depth_pixel_selector.valid_selected_idx,
		m_clip_img_rows, m_clip_img_cols,
		valid_surfel_array,
		stream
	);
}


/**
* The methods for foreground segmentation
*/
void surfelwarp::ImageProcessor::allocateForegroundSegmentationBuffer() {
	//Create the segmenter
	const auto& config = ConfigParser::Instance();
	if(config.use_offline_foreground_segmneter()) {
		LOG(INFO) << "Use pre-computed segmentation mask";
		m_foreground_segmenter = std::make_shared<ForegroundSegmenterOffline>();
	}
	else {
		m_foreground_segmenter = std::make_shared<ForegroundSegmenterPermutohedral>();
		//m_foreground_segmenter = std::make_shared<ForegroundSegmenterNaiveCRF>();
	}
	
	//Allocate the buffer for them segmenter
	m_foreground_segmenter->AllocateBuffer(m_clip_img_rows, m_clip_img_cols);
}

void surfelwarp::ImageProcessor::releaseForegroundSegmentationBuffer() {
	m_foreground_segmenter->ReleaseBuffer();
}

void surfelwarp::ImageProcessor::SegmentForeground(int frame_idx, cudaStream_t stream) {
	//Hand on the input to the segmenter
	m_foreground_segmenter->SetInputImages(
		m_clip_normalize_rgb_collect.texture,
		m_depth_raw_collect.texture,
		m_depth_filter_collect.texture,
		frame_idx
	);
	
	//Invoke it
	m_foreground_segmenter->Segment(stream);
}


/* Method for sparse feature correspondence
 */
void surfelwarp::ImageProcessor::allocateFeatureCorrespondenceBuffer() {
	//Use gpc?
	m_feature_correspondence_finder = std::make_shared<PatchColliderRGBCorrespondence>();
	
	//Allocate the buffer for it
	m_feature_correspondence_finder->AllocateBuffer(clip_rows(), clip_cols());
}

void surfelwarp::ImageProcessor::releaseFeatureCorrespondenceBuffer() {
	m_feature_correspondence_finder->ReleaseBuffer();
}

void surfelwarp::ImageProcessor::FindCorrespondence(cudaStream_t stream) {
	//Provide the input
	m_feature_correspondence_finder->SetInputImages(
		ClipNormalizedRGBTexturePrev(),
		ClipNormalizedRGBTexture(),
		ForegroundMask()
	);
	
	//Find the correspondence
	m_feature_correspondence_finder->FindCorrespondence(stream);
}


/* The method for compute the gradient map
 */
void surfelwarp::ImageProcessor::allocateGradientMap()
{
	createFloat2TextureSurface(m_clip_img_rows, m_clip_img_cols, m_foreground_mask_gradient_map_collect);
	createFloat2TextureSurface(m_clip_img_rows, m_clip_img_cols, m_density_gradient_map_collect);
}

void surfelwarp::ImageProcessor::releaseGradientMap()
{
	releaseTextureCollect(m_foreground_mask_gradient_map_collect);
	releaseTextureCollect(m_density_gradient_map_collect);
}


void surfelwarp::ImageProcessor::ComputeGradientMap(cudaStream_t stream)
{
	constexpr bool kUseFilteredMap = false;
	if(kUseFilteredMap)
	{
		computeDensityForegroundMaskGradient(
			FilterForegroundMask(),
			m_filter_density_map_collect.texture,
			m_clip_img_rows, m_clip_img_cols,
			m_foreground_mask_gradient_map_collect.surface,
			m_density_gradient_map_collect.surface,
			stream
		);
	}
	else
	{
		computeDensityForegroundMaskGradient(
			FilterForegroundMask(),
			m_density_map_collect.texture, 
			m_clip_img_rows, m_clip_img_cols, 
			m_foreground_mask_gradient_map_collect.surface, 
			m_density_gradient_map_collect.surface, 
			stream
		);
	}
}
