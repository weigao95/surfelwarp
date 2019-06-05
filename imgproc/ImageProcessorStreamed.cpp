#include "imgproc/ImageProcessor.h"

void surfelwarp::ImageProcessor::initProcessorStream() {
	//Create the stream
	cudaSafeCall(cudaStreamCreate(&m_processor_stream[0]));
	cudaSafeCall(cudaStreamCreate(&m_processor_stream[1]));
	cudaSafeCall(cudaStreamCreate(&m_processor_stream[2]));
}

void surfelwarp::ImageProcessor::releaseProcessorStream() {
	//Destroy these streams
	cudaSafeCall(cudaStreamDestroy(m_processor_stream[0]));
	cudaSafeCall(cudaStreamDestroy(m_processor_stream[1]));
	cudaSafeCall(cudaStreamDestroy(m_processor_stream[2]));

	//Assign to null value
	m_processor_stream[0] = 0;
	m_processor_stream[1] = 0;
	m_processor_stream[2] = 0;
}

void surfelwarp::ImageProcessor::syncAllProcessorStream() {
	cudaSafeCall(cudaStreamSynchronize(m_processor_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_processor_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_processor_stream[2]));
}

void surfelwarp::ImageProcessor::ProcessFrameStreamed(CameraObservation & observation, size_t frame_idx) {
	FetchFrame(frame_idx);
	UploadDepthImage(m_processor_stream[0]);
	UploadRawRGBImage(m_processor_stream[0]);

	//This seems cause some problem ,disable it at first
	//ReprojectDepthToRGB(stream);

	ClipFilterDepthImage(m_processor_stream[0]);
	ClipNormalizeRGBImage(m_processor_stream[0]);

	//The geometry map
	BuildVertexConfigMap(m_processor_stream[0]);
	BuildNormalRadiusMap(m_processor_stream[0]);
	BuildColorTimeTexture(frame_idx, m_processor_stream[0]);

	//Sync here
	cudaSafeCall(cudaStreamSynchronize(m_processor_stream[0]));

	//Invoke other expensive computations
	SegmentForeground(frame_idx, m_processor_stream[0]); //This doesn't block, even for hashing based method
	FindCorrespondence(m_processor_stream[1]); //This will block, thus sync inside

	//The gradient map depends on filtered mask
	cudaSafeCall(cudaStreamSynchronize(m_processor_stream[0]));
	ComputeGradientMap(m_processor_stream[0]);

	//Sync and output
	syncAllProcessorStream();
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
	observation.density_gradient_map = DensityGradientTexture();

	//The foreground masks
	observation.foreground_mask = ForegroundMask();
	observation.filter_foreground_mask = FilterForegroundMask();
	observation.foreground_mask_gradient_map = ForegroundMaskGradientTexture();

	//The correspondence pixel pairs
	const auto& pixel_pair_array = CorrespondencePixelPair();
	observation.correspondence_pixel_pairs = DeviceArrayView<ushort4>(pixel_pair_array.ptr(), pixel_pair_array.size());
}