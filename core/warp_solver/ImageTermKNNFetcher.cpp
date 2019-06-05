#include "common/ConfigParser.h"
#include "core/warp_solver/ImageTermKNNFetcher.h"

surfelwarp::ImageTermKNNFetcher::ImageTermKNNFetcher() {
	//The initialization part
	const auto& config = ConfigParser::Instance();
	m_image_height = config.clip_image_rows();
	m_image_width = config.clip_image_cols();
	memset(&m_geometry_maps, 0, sizeof(m_geometry_maps));
	
	//The malloc part
	const auto num_pixels = m_image_height * m_image_width;
	m_potential_pixel_indicator.create(num_pixels);

	//For compaction
	m_indicator_prefixsum.InclusiveSum(num_pixels);
	m_potential_pixels.AllocateBuffer(num_pixels);
	m_dense_image_knn.AllocateBuffer(num_pixels);
	m_dense_image_knn_weight.AllocateBuffer(num_pixels);

	//The page-locked memory
	cudaSafeCall(cudaMallocHost((void**)&m_num_potential_pixel, sizeof(unsigned)));
}

surfelwarp::ImageTermKNNFetcher::~ImageTermKNNFetcher() {
	m_potential_pixel_indicator.release();

	m_potential_pixels.ReleaseBuffer();
	m_dense_image_knn.ReleaseBuffer();
	m_dense_image_knn_weight.ReleaseBuffer();

	cudaSafeCall(cudaFreeHost(m_num_potential_pixel));
}

void surfelwarp::ImageTermKNNFetcher::SetInputs(
	const DeviceArrayView2D<surfelwarp::KNNAndWeight> &knn_map,
	cudaTextureObject_t index_map
) {
	m_geometry_maps.knn_map = knn_map;
	m_geometry_maps.index_map = index_map;
}


//Methods for sanity check
void surfelwarp::ImageTermKNNFetcher::CheckDenseImageTermKNN(const surfelwarp::DeviceArrayView<ushort4> &dense_depth_knn_gpu) {
	LOG(INFO) << "Check the image term knn against dense depth knn";
	
	//Should be called after sync
	SURFELWARP_CHECK_EQ(m_dense_image_knn.ArraySize(), m_potential_pixels.ArraySize());
	SURFELWARP_CHECK_EQ(m_dense_image_knn.ArraySize(), m_dense_image_knn_weight.ArraySize());
	SURFELWARP_CHECK_EQ(m_dense_image_knn.ArraySize(), dense_depth_knn_gpu.Size());
	
	//Download the data
	std::vector<ushort4> potential_pixel_knn_array, dense_depth_knn_array;
	dense_depth_knn_gpu.Download(dense_depth_knn_array);
	m_dense_image_knn.ArrayReadOnly().Download(potential_pixel_knn_array);
	
	//Iterates
	for(auto i = 0; i < dense_depth_knn_array.size(); i++) {
		const auto& pixel_knn = potential_pixel_knn_array[i];
		const auto& depth_knn = dense_depth_knn_array[i];
		SURFELWARP_CHECK(pixel_knn.x == depth_knn.x);
		SURFELWARP_CHECK(pixel_knn.y == depth_knn.y);
		SURFELWARP_CHECK(pixel_knn.z == depth_knn.z);
		SURFELWARP_CHECK(pixel_knn.w == depth_knn.w);
	}
	
	//Seems correct
	LOG(INFO) << "Check done! Seems correct!";
}

