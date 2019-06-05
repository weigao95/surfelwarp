#include "common/ConfigParser.h"
#include "core/warp_solver/DensityForegroundMapHandler.h"

surfelwarp::DensityForegroundMapHandler::DensityForegroundMapHandler() {
	const auto& config = ConfigParser::Instance();
	m_image_height = config.clip_image_rows();
	m_image_width = config.clip_image_cols();
	m_project_intrinsic = config.rgb_intrinsic_clip();
	
	memset(&m_depth_observation, 0, sizeof(m_depth_observation));
	memset(&m_geometry_maps, 0, sizeof(m_geometry_maps));
}

void surfelwarp::DensityForegroundMapHandler::AllocateBuffer() {
	const auto num_pixels = m_image_height * m_image_width;
	//The buffer of the marked pixel pairs
	m_color_pixel_indicator_map.create(num_pixels);
	m_mask_pixel_indicator_map.create(num_pixels);
	
	//The compaction maps
	m_color_pixel_indicator_prefixsum.AllocateBuffer(num_pixels);
	m_mask_pixel_indicator_prefixsum.AllocateBuffer(num_pixels);
	m_valid_color_pixel.AllocateBuffer(num_pixels);
	m_valid_mask_pixel.AllocateBuffer(num_pixels);
	m_valid_color_pixel_knn.AllocateBuffer(num_pixels);
	m_valid_mask_pixel_knn.AllocateBuffer(num_pixels);
	m_valid_color_pixel_knn_weight.AllocateBuffer(num_pixels);
	m_valid_mask_pixel_knn_weight.AllocateBuffer(num_pixels);

	//The page-locked memory
	cudaSafeCall(cudaMallocHost((void**)(&m_num_mask_pixel), sizeof(unsigned)));
	
	//The twist gradient
	m_color_residual.AllocateBuffer(num_pixels);
	m_color_twist_gradient.AllocateBuffer(num_pixels);
	m_foreground_residual.AllocateBuffer(num_pixels);
	m_foreground_twist_gradient.AllocateBuffer(num_pixels);
}

void surfelwarp::DensityForegroundMapHandler::ReleaseBuffer() {
	m_color_pixel_indicator_map.release();
	m_mask_pixel_indicator_map.release();
	
	//The compaction maps
	m_valid_color_pixel.ReleaseBuffer();
	m_valid_mask_pixel.ReleaseBuffer();
	m_valid_color_pixel_knn.ReleaseBuffer();
	m_valid_mask_pixel_knn.ReleaseBuffer();
	m_valid_color_pixel_knn_weight.ReleaseBuffer();
	m_valid_mask_pixel_knn_weight.ReleaseBuffer();

	//The page-lock memory
	cudaSafeCall(cudaFreeHost(m_num_mask_pixel));
	
	//The twist gradient
	m_color_residual.ReleaseBuffer();
	m_color_twist_gradient.ReleaseBuffer();
	m_foreground_residual.ReleaseBuffer();
	m_foreground_twist_gradient.ReleaseBuffer();
}

/* The processing interface
 */
void surfelwarp::DensityForegroundMapHandler::SetInputs(
	const DeviceArrayView<DualQuaternion>& node_se3,
	const DeviceArrayView2D<KNNAndWeight>& knn_map,
	cudaTextureObject_t foreground_mask, 
	cudaTextureObject_t filtered_foreground_mask,
	cudaTextureObject_t foreground_gradient_map,
	//The color density terms
	cudaTextureObject_t density_map,
	cudaTextureObject_t density_gradient_map,
	//The rendered maps
	cudaTextureObject_t reference_vertex_map,
	cudaTextureObject_t reference_normal_map,
	cudaTextureObject_t index_map,
	cudaTextureObject_t normalized_rgb_map,
	const mat34& world2camera,
	//The potential pixels,
	const ImageTermKNNFetcher::ImageTermPixelAndKNN& potential_pixels_knn
) {
	m_node_se3 = node_se3;
	m_knn_map = knn_map;
	m_world2camera = world2camera;
	
	m_depth_observation.foreground_mask = foreground_mask;
	m_depth_observation.filtered_foreground_mask = filtered_foreground_mask;
	m_depth_observation.foreground_mask_gradient_map = foreground_gradient_map;
	m_depth_observation.density_map = density_map;
	m_depth_observation.density_gradient_map = density_gradient_map;
	
	m_geometry_maps.reference_vertex_map = reference_vertex_map;
	m_geometry_maps.reference_normal_map = reference_normal_map;
	m_geometry_maps.index_map = index_map;
	m_geometry_maps.normalized_rgb_map = normalized_rgb_map;
	
	m_potential_pixels_knn = potential_pixels_knn;
}


void surfelwarp::DensityForegroundMapHandler::UpdateNodeSE3(DeviceArrayView<DualQuaternion> node_se3) {
	SURFELWARP_CHECK_EQ(node_se3.Size(), m_node_se3.Size());
	m_node_se3 = node_se3;
}

void surfelwarp::DensityForegroundMapHandler::FindValidColorForegroundMaskPixels(
	cudaStream_t color_stream,
	cudaStream_t mask_stream
) {
	//Use color stream for marking the value
	MarkValidColorForegroundMaskPixels(color_stream);
	
	//Sync before using more streams
	cudaSafeCall(cudaStreamSynchronize(color_stream));
	
	//Use two streams for compaction
	CompactValidColorPixel(color_stream);
	CompactValidMaskPixel(mask_stream);

	//Query the size: this will sync the stream, these will sync
	QueryCompactedColorPixelArraySize(color_stream);
	QueryCompactedMaskPixelArraySize(mask_stream);
}

void surfelwarp::DensityForegroundMapHandler::FindPotentialForegroundMaskPixelSynced(cudaStream_t stream) {
	//Use color stream for marking the value
	MarkValidColorForegroundMaskPixels(stream);

	//Use two streams for compaction
	CompactValidMaskPixel(stream);

	//Query the size: this will sync the stream, these will sync
	QueryCompactedMaskPixelArraySize(stream);
}
