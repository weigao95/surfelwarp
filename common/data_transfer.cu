#include "common/data_transfer.h"
#include "common/common_utils.h"
#include "common/encode_utils.h"
#include "common/logging.h"
#include "common/common_texture_utils.h"
#include "common/common_point_cloud_utils.h"
#include "math/vector_ops.hpp"
#include <assert.h>
#include <Eigen/Eigen>
#include <device_launch_parameters.h>


cv::Mat surfelwarp::downloadDepthImage(const DeviceArray2D<unsigned short>& image_gpu) {
	const auto num_rows = image_gpu.rows();
	const auto num_cols = image_gpu.cols();
	cv::Mat depth_cpu(num_rows, num_cols, CV_16UC1);
	image_gpu.download(depth_cpu.data, sizeof(unsigned short) * num_cols);
	return depth_cpu;
}

cv::Mat surfelwarp::downloadDepthImage(cudaTextureObject_t image_gpu) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(image_gpu, width, height);
	DeviceArray2D<unsigned short> map;
	map.create(height, width);

	//Transfer and download
	textureToMap2D<unsigned short>(image_gpu, map);
	return downloadDepthImage(map);
}

cv::Mat surfelwarp::downloadRGBImage(
		const DeviceArray<uchar3>& image_gpu,
		const unsigned rows, const unsigned cols
) {
	assert(rows * cols == image_gpu.size());
	cv::Mat rgb_cpu(rows, cols, CV_8UC3);
	image_gpu.download((uchar3*) (rgb_cpu.data));
	return rgb_cpu;
}

cv::Mat surfelwarp::downloadNormalizeRGBImage(const DeviceArray2D<float4>& rgb_img) {
	cv::Mat rgb_cpu(rgb_img.rows(), rgb_img.cols(), CV_32FC4);
	rgb_img.download(rgb_cpu.data, sizeof(float4) * rgb_img.cols());
	return rgb_cpu;
}

cv::Mat surfelwarp::downloadNormalizeRGBImage(cudaTextureObject_t rgb_img) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(rgb_img, width, height);
	DeviceArray2D<float4> map;
	map.create(height, width);

	//Transfer and download
	textureToMap2D<float4>(rgb_img, map);
	return downloadNormalizeRGBImage(map);
}

cv::Mat surfelwarp::rgbImageFromColorTimeMap(cudaTextureObject_t color_time_map) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(color_time_map, width, height);

	//First download to device array
	DeviceArray2D<float4> map;
	map.create(height, width);
	textureToMap2D<float4>(color_time_map, map);

	//Donwload to host
	std::vector<float4> color_time_host;
	int cols = width;
	map.download(color_time_host, cols);

	cv::Mat rgb_cpu(height, width, CV_8UC3);
	for (auto i = 0; i < width; i++) {
		for (auto j = 0; j < height; j++) {
			const auto flatten_idx = i + j * width;
			const float4 color_time_value = color_time_host[flatten_idx];
			uchar3 rgb_value;
			float_decode_rgb(color_time_value.x, rgb_value);
			rgb_cpu.at<unsigned char>(j, sizeof(uchar3) * i + 0) = rgb_value.x;
			rgb_cpu.at<unsigned char>(j, sizeof(uchar3) * i + 1) = rgb_value.y;
			rgb_cpu.at<unsigned char>(j, sizeof(uchar3) * i + 2) = rgb_value.z;
		}
	}
	return rgb_cpu;
}

cv::Mat surfelwarp::normalMapForVisualize(cudaTextureObject_t normal_map) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(normal_map, width, height);

	//First download to device array
	DeviceArray2D<float4> map;
	map.create(height, width);
	textureToMap2D<float4>(normal_map, map);

	//Donwload to host
	std::vector<float4> normal_map_host;
	int cols = width;
	map.download(normal_map_host, cols);

	cv::Mat rgb_cpu(height, width, CV_8UC3);
	for (auto i = 0; i < width; i++) {
		for (auto j = 0; j < height; j++) {
			const auto flatten_idx = i + j * width;
			const float4 normal_value = normal_map_host[flatten_idx];
			uchar3 rgb_value;
			rgb_value.x = (unsigned char) ((normal_value.x + 1) * 120.0f);
			rgb_value.y = (unsigned char) ((normal_value.y + 1) * 120.0f);
			rgb_value.z = (unsigned char) ((normal_value.z + 1) * 120.0f);
			rgb_cpu.at<unsigned char>(j, sizeof(uchar3) * i + 0) = rgb_value.x;
			rgb_cpu.at<unsigned char>(j, sizeof(uchar3) * i + 1) = rgb_value.y;
			rgb_cpu.at<unsigned char>(j, sizeof(uchar3) * i + 2) = rgb_value.z;
		}
	}
	return rgb_cpu;
}

void surfelwarp::downloadSegmentationMask(cudaTextureObject_t mask, std::vector<unsigned char>& h_mask) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(mask, width, height);

	//Download it to device
	DeviceArray2D<unsigned char> d_mask;
	d_mask.create(height, width);
	textureToMap2D<unsigned char>(mask, d_mask);

	//Download it to host
	int h_cols;
	d_mask.download(h_mask, h_cols);
}

cv::Mat surfelwarp::downloadRawSegmentationMask(cudaTextureObject_t mask) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(mask, width, height);

	//Download it to device
	DeviceArray2D<unsigned char> d_mask;
	d_mask.create(height, width);
	textureToMap2D<unsigned char>(mask, d_mask);

	//Download it to host
	std::vector<unsigned char> h_mask_vec;
	int h_cols;
	d_mask.download(h_mask_vec, h_cols);

	cv::Mat raw_mask(height, width, CV_8UC1);
	for (auto row = 0; row < height; row++) {
		for (auto col = 0; col < width; col++) {
			const auto offset = col + row * width;
			raw_mask.at<unsigned char>(row, col) = h_mask_vec[offset];
		}
	}

	return raw_mask;
}

void surfelwarp::downloadGrayScaleImage(cudaTextureObject_t image, cv::Mat& h_image, float scale) {
	//Query the size of texture
	unsigned width = 0, height = 0;
	query2DTextureExtent(image, width, height);

	//Download it to device
	DeviceArray2D<float> d_meanfield;
	d_meanfield.create(height, width);
	textureToMap2D<float>(image, d_meanfield);

	//To host
	cv::Mat h_meanfield_prob = cv::Mat(height, width, CV_32FC1);
	d_meanfield.download(h_meanfield_prob.data, sizeof(float) * width);

	//Transfer it
	h_meanfield_prob.convertTo(h_image, CV_8UC1, scale * 255.f);
}


void surfelwarp::downloadTransferBinaryMeanfield(cudaTextureObject_t meanfield_q, cv::Mat& h_meanfield_uchar) {
	downloadGrayScaleImage(meanfield_q, h_meanfield_uchar);
}

/* The point cloud downloading method
 */
PointCloud3f_Pointer surfelwarp::downloadPointCloud(const surfelwarp::DeviceArray<float4>& vertex) {
	PointCloud3f_Pointer point_cloud(new PointCloud3f);
	std::vector<float4> h_vertex;
	vertex.download(h_vertex);
	setPointCloudSize(point_cloud, vertex.size());
	for (auto idx = 0; idx < vertex.size(); idx++) {
		setPoint(h_vertex[idx].x, h_vertex[idx].y, h_vertex[idx].z, point_cloud, idx);
	}
	return point_cloud;
}

PointCloud3f_Pointer surfelwarp::downloadPointCloud(const DeviceArray2D<float4>& vertex_map) {
	PointCloud3f_Pointer point_cloud(new PointCloud3f);
	const auto num_rows = vertex_map.rows();
	const auto num_cols = vertex_map.cols();
	const auto total_size = num_cols * num_rows;
	float4* host_ptr = new float4[total_size];
	vertex_map.download(host_ptr, num_cols * sizeof(float4));
	size_t valid_count = 0;
	setPointCloudSize(point_cloud, total_size);
	for (int idx = 0; idx < total_size; idx += 1) {
		float x = host_ptr[idx].x;
		float y = host_ptr[idx].y;
		float z = host_ptr[idx].z;
		if (std::abs(x > 1e-3) || std::abs(y > 1e-3) || std::abs(z > 1e-3)) {
			valid_count++;
		}
		setPoint(x, y, z, point_cloud, idx);
	}
	//LOG(INFO) << "The number of valid point cloud is " << valid_count << std::endl;
	delete[] host_ptr;
	return point_cloud;
}

PointCloud3f_Pointer surfelwarp::downloadPointCloud(
		const DeviceArray2D<float4>& vertex_map,
		DeviceArrayView<unsigned int> indicator) {
	PointCloud3f_Pointer point_cloud(new PointCloud3f);
	const auto num_rows = vertex_map.rows();
	const auto num_cols = vertex_map.cols();
	const auto total_size = num_cols * num_rows;
	float4* host_ptr = new float4[total_size];
	vertex_map.download(host_ptr, num_cols * sizeof(float4));

	std::vector<unsigned> h_indicator;
	indicator.Download(h_indicator);
#ifdef WITH_CILANTRO
	int valid_point_count = 0;
	for (int idx = 0; idx < total_size; idx += 1) {
		if (h_indicator[idx]) valid_point_count++;
	}
	setPointCloudSize(point_cloud, valid_point_count);
#endif

	for (int idx = 0; idx < total_size; idx += 1) {
		if (h_indicator[idx]) {
			setPoint(host_ptr[idx].x, host_ptr[idx].y, host_ptr[idx].z, point_cloud, idx);
		}
	}
	//LOG(INFO) << "The number of valid point cloud is " << valid_count << std::endl;
	delete[] host_ptr;
	return point_cloud;
}

PointCloud3f_Pointer surfelwarp::downloadPointCloud(
		const DeviceArray2D<float4>& vertex_map,
		DeviceArrayView<ushort2> pixel
) {
	PointCloud3f_Pointer point_cloud(new PointCloud3f);
	const auto num_rows = vertex_map.rows();
	const auto num_cols = vertex_map.cols();
	const auto total_size = num_cols * num_rows;
	float4* host_ptr = new float4[total_size];
	vertex_map.download(host_ptr, num_cols * sizeof(float4));
	std::vector<ushort2> h_pixels;
	pixel.Download(h_pixels);
	setPointCloudSize(point_cloud, h_pixels.size());
	for (auto i = 0; i < h_pixels.size(); i++) {
		const auto idx = h_pixels[i].x + h_pixels[i].y * vertex_map.cols();
		setPoint(host_ptr[idx].x, host_ptr[idx].y, host_ptr[idx].z, point_cloud, i);
	}
	delete[] host_ptr;
	return point_cloud;
}

void surfelwarp::downloadPointCloud(const DeviceArray2D<float4>& vertex_map, std::vector<float4>& point_cloud) {
	point_cloud.clear();
	const auto num_rows = vertex_map.rows();
	const auto num_cols = vertex_map.cols();
	const auto total_size = num_cols * num_rows;
	float4* host_ptr = new float4[total_size];
	vertex_map.download(host_ptr, num_cols * sizeof(float4));
	for (int idx = 0; idx < total_size; idx += 1) {
		float4 point;
		point.x = host_ptr[idx].x;
		point.y = host_ptr[idx].y;
		point.z = host_ptr[idx].z;
		if (std::abs(point.x > 1e-3) || std::abs(point.y > 1e-3) || std::abs(point.z > 1e-3))
			point_cloud.push_back(point);
	}
	delete[] host_ptr;
}

PointCloud3f_Pointer surfelwarp::downloadPointCloud(cudaTextureObject_t vertex_map) {
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array;
	vertex_map_array.create(rows, cols);
	textureToMap2D<float4>(vertex_map, vertex_map_array);
	return downloadPointCloud(vertex_map_array);
}

PointCloud3f_Pointer
surfelwarp::downloadPointCloud(cudaTextureObject_t vertex_map, DeviceArrayView<unsigned int> indicator) {
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array;
	vertex_map_array.create(rows, cols);
	textureToMap2D<float4>(vertex_map, vertex_map_array);
	return downloadPointCloud(vertex_map_array, indicator);
}

PointCloud3f_Pointer
surfelwarp::downloadPointCloud(cudaTextureObject_t vertex_map, DeviceArrayView<ushort2> pixel) {
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array;
	vertex_map_array.create(rows, cols);
	textureToMap2D<float4>(vertex_map, vertex_map_array);
	return downloadPointCloud(vertex_map_array, pixel);
}

void surfelwarp::downloadPointCloud(cudaTextureObject_t vertex_map, std::vector<float4>& point_cloud) {
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array;
	vertex_map_array.create(rows, cols);
	textureToMap2D<float4>(vertex_map, vertex_map_array);
	downloadPointCloud(vertex_map_array, point_cloud);
}

#ifdef WITH_PCL
PointCloudNormal_Pointer surfelwarp::downloadNormalCloud(const DeviceArray<float4>& d_normal) {
	std::vector<float4> h_normal;
	d_normal.download(h_normal);
	PointCloudNormal_Pointer normal_cloud(new PointCloudNormal);
	for (auto idx = 0; idx < d_normal.size(); idx++) {
		setNormal(h_normal[idx].x, h_normal[idx].y, h_normal[idx].z, normal_cloud, idx);
	}
	return normal_cloud;
}
#elif defined(WITH_CILANTRO)

void surfelwarp::downloadNormalCloud(const DeviceArray<float4>& d_normal, PointCloudNormal_Pointer& point_cloud) {
	std::vector<float4> h_normal;
	d_normal.download(h_normal);
	setNormalCloudSize(point_cloud, d_normal.size());
	for (auto idx = 0; idx < d_normal.size(); idx++) {
		setNormal(h_normal[idx].x, h_normal[idx].y, h_normal[idx].z, point_cloud, idx);
	}
}

#endif

#ifdef WITH_PCL
PointCloudNormal_Pointer surfelwarp::downloadNormalCloud(const DeviceArray2D<float4>& normal_map) {
	PointCloudNormal_Pointer normal_cloud(new PointCloudNormal);
	const auto num_rows = normal_map.rows();
	const auto num_cols = normal_map.cols();
	const auto total_size = num_cols * num_rows;
	float4* host_ptr = new float4[total_size];
	normal_map.download(host_ptr, num_cols * sizeof(float4));
	int valid_count = 0;
	for (int idx = 0; idx < total_size; idx += 1) {
		float4 normal_dev = host_ptr[idx];
		SURFELWARP_CHECK(!isnan(normal_dev.x));
		SURFELWARP_CHECK(!isnan(normal_dev.y));
		SURFELWARP_CHECK(!isnan(normal_dev.z));
		if (norm(make_float3(host_ptr[idx].x, host_ptr[idx].y, host_ptr[idx].z)) > 1e-4) {
			valid_count++;
		}
		setNormal(normal_dev.x, normal_dev.y, normal_dev.z, normal_cloud, idx);
	}
	//LOG(INFO) << "The number of valid normals is " << valid_count;
	delete[] host_ptr;
	return normal_cloud;
}
#elif defined(WITH_CILANTRO)

void surfelwarp::downloadNormalCloud(const DeviceArray2D<float4>& normal_map, PointCloudNormal_Pointer& point_cloud) {
	const auto num_rows = normal_map.rows();
	const auto num_cols = normal_map.cols();
	const auto total_size = num_cols * num_rows;
	float4* host_ptr = new float4[total_size];
	normal_map.download(host_ptr, num_cols * sizeof(float4));
	int valid_count = 0;
	setNormalCloudSize(point_cloud, total_size);
	for (int idx = 0; idx < total_size; idx += 1) {
		float4 normal_dev = host_ptr[idx];
		SURFELWARP_CHECK(!isnan(normal_dev.x));
		SURFELWARP_CHECK(!isnan(normal_dev.y));
		SURFELWARP_CHECK(!isnan(normal_dev.z));
		if (norm(make_float3(host_ptr[idx].x, host_ptr[idx].y, host_ptr[idx].z)) > 1e-4) {
			valid_count++;
		}
		setNormal(normal_dev.x, normal_dev.y, normal_dev.z, point_cloud, idx);
	}
	//LOG(INFO) << "The number of valid normals is " << valid_count;
	delete[] host_ptr;
}

#endif

#ifdef WITH_PCL
pcl::PointCloud<pcl::Normal>::Ptr surfelwarp::downloadNormalCloud(cudaTextureObject_t normal_map) {
	unsigned rows, cols;
	query2DTextureExtent(normal_map, cols, rows);
	DeviceArray2D<float4> normal_map_array;
	normal_map_array.create(rows, cols);
	textureToMap2D<float4>(normal_map, normal_map_array);
	return downloadNormalCloud(normal_map_array);
}
#elif defined(WITH_CILANTRO)

void surfelwarp::downloadNormalCloud(cudaTextureObject_t normal_map, PointCloudNormal_Pointer& point_cloud) {
	unsigned rows, cols;
	query2DTextureExtent(normal_map, cols, rows);
	DeviceArray2D<float4> normal_map_array;
	normal_map_array.create(rows, cols);
	textureToMap2D<float4>(normal_map, normal_map_array);
	downloadNormalCloud(normal_map_array, point_cloud);
}

#endif

void surfelwarp::downloadPointNormalCloud(
		const surfelwarp::DeviceArray<DepthSurfel>& surfel_array,
		PointCloud3f_Pointer& point_cloud,
#ifdef WITH_PCL
		PointCloudNormal_Pointer& normal_cloud,
#endif
		const float point_scale
) {
	//Prepare the data
	point_cloud = PointCloud3f_Pointer(new PointCloud3f);
#ifdef WITH_PCL
	normal_cloud = PointCloudNormal_Pointer(new PointCloudNormal);
#elif defined(WITH_CILANTRO)
	// in cilantro, the normals are a field within the point cloud, we don't need a separate cloud
	auto& normal_cloud = point_cloud;
#endif

	//Download it
	std::vector<DepthSurfel> surfel_array_host;
	surfel_array.download(surfel_array_host);

	setPointCloudSize(point_cloud, surfel_array_host.size());
	setNormalCloudSize(normal_cloud, surfel_array_host.size());

	//Construct the output
	for (auto i = 0; i < surfel_array_host.size(); i++) {
		DepthSurfel surfel = surfel_array_host[i];
		setPoint(surfel.vertex_confid.x, surfel.vertex_confid.y, surfel.vertex_confid.z, point_cloud, i, point_scale);
		setNormal(surfel.normal_radius.x, surfel.normal_radius.y, surfel.normal_radius.z, normal_cloud, i);
	}
}

void surfelwarp::separateDownloadPointCloud(const surfelwarp::DeviceArrayView<float4>& point_cloud,
                                            const surfelwarp::DeviceArrayView<unsigned int>& indicator,
                                            PointCloud3f_Pointer& fused_cloud,
                                            PointCloud3f_Pointer& unfused_cloud) {
	std::vector<float4> h_surfels;
	std::vector<unsigned> h_indicator;
	point_cloud.Download(h_surfels);
	indicator.Download(h_indicator);
	SURFELWARP_CHECK(h_indicator.size() == h_surfels.size());
#ifdef WITH_CILANTRO
	int fused_cloud_size = 0;
	int unfused_cloud_size = 0;
	for (auto i = 0; i < h_surfels.size(); i++) {
		const auto indicator = h_indicator[i];
		if (indicator > 0) {
			fused_cloud_size++;
		} else {
			unfused_cloud_size++;
		}
	}
	setPointCloudSize(fused_cloud, fused_cloud_size);
	setPointCloudSize(unfused_cloud, unfused_cloud_size);
#endif
	int i_fused = 0;
	int i_unfused = 0;
	for (auto i = 0; i < h_surfels.size(); i++) {
		const auto indicator = h_indicator[i];
		const auto flat_point = h_surfels[i];
		if (indicator > 0) {
			setPoint(flat_point.x, flat_point.y, flat_point.z, fused_cloud, i_fused);
			i_fused++;
		} else {
			setPoint(flat_point.x, flat_point.y, flat_point.z, unfused_cloud, i_unfused);
			i_unfused++;
		}
	}
}

void surfelwarp::separateDownloadPointCloud(
		const surfelwarp::DeviceArrayView<float4>& point_cloud,
		unsigned num_remaining_surfels,
		PointCloud3f_Pointer& remaining_cloud,
		PointCloud3f_Pointer& appended_cloud
) {
	//Clear the existing point cloud
#ifdef WITH_PCL
	remaining_cloud->points.clear();
	appended_cloud->points.clear();
#endif
	setPointCloudSize(remaining_cloud, num_remaining_surfels);
	setPointCloudSize(appended_cloud, point_cloud.Size() - num_remaining_surfels);

	std::vector<float4> h_surfels;
	point_cloud.Download(h_surfels);
	int i_appended = 0;
	for (auto i = 0; i < point_cloud.Size(); i++) {
		const auto flat_point = h_surfels[i];
		if (i < num_remaining_surfels) {
			setPoint(flat_point.x, flat_point.y, flat_point.z, remaining_cloud, i);
		} else {
			setPoint(flat_point.x, flat_point.y, flat_point.z, appended_cloud, i_appended);
			i_appended++;
		}
	}
}


/* The download function for colored point cloud
 */
PointCloud3fRGB_Pointer
surfelwarp::downloadColoredPointCloud(
		const surfelwarp::DeviceArray<float4>& vertex_confid,
		const surfelwarp::DeviceArray<float4>& color_time
) {
	PointCloud3fRGB_Pointer point_cloud(new PointCloud3fRGB());
	std::vector<float4> h_vertex, h_color_time;
	vertex_confid.download(h_vertex);
	color_time.download(h_color_time);
	SURFELWARP_CHECK_EQ(h_vertex.size(), h_color_time.size());
	setPointCloudRGBSize(point_cloud, h_vertex.size());
	for (auto idx = 0; idx < h_vertex.size(); idx++) {
		float encoded_rgb = h_color_time[idx].x;
		uchar3 rgb;
		float_decode_rgb(encoded_rgb, rgb);
		setPointRGB(h_vertex[idx].x, h_vertex[idx].y, h_vertex[idx].z,
		            rgb.x, rgb.y, rgb.z,
		            point_cloud, idx);
	}
	return point_cloud;
}


PointCloud3fRGB_Pointer
surfelwarp::downloadColoredPointCloud(
		cudaTextureObject_t vertex_map,
		cudaTextureObject_t color_time_map,
		bool flip_color
) {
	unsigned rows, cols;
	query2DTextureExtent(vertex_map, cols, rows);
	DeviceArray2D<float4> vertex_map_array, color_map_array;
	vertex_map_array.create(rows, cols);
	color_map_array.create(rows, cols);
	textureToMap2D<float4>(vertex_map, vertex_map_array);
	textureToMap2D<float4>(color_time_map, color_map_array);

	//Download it
	float4* h_vertex = new float4[rows * cols];
	float4* h_color_time = new float4[rows * cols];
	vertex_map_array.download(h_vertex, cols * sizeof(float4));
	color_map_array.download(h_color_time, cols * sizeof(float4));

	//Construct the point cloud
	PointCloud3fRGB_Pointer point_cloud(new PointCloud3fRGB());
	setPointCloudRGBSize(point_cloud, rows * cols);
	for (auto i = 0; i < rows * cols; i++) {
		float encoded_rgb = h_color_time[i].x;
		uchar3 rgb;
		float_decode_rgb(encoded_rgb, rgb);

		if (flip_color) {
			setPointRGB(h_vertex[i].x, h_vertex[i].y, h_vertex[i].z,
			            rgb.z, rgb.y, rgb.x,
			            point_cloud, i);
		} else {
			setPointRGB(h_vertex[i].x, h_vertex[i].y, h_vertex[i].z,
			            rgb.x, rgb.y, rgb.z,
			            point_cloud, i);
		}
	}

	delete[] h_vertex;
	delete[] h_color_time;
	return point_cloud;
}


//The method to add color to point cloud
PointCloud3fRGB_Pointer surfelwarp::addColorToPointCloud(
		const PointCloud3f_Pointer& point_cloud,
		uchar4 rgba
) {
	PointCloud3fRGB_Pointer color_cloud(new PointCloud3fRGB());
	setPointCloudRGBSize(color_cloud, point_cloud->size());
	for (auto i = 0; i < point_cloud->size(); i++) {
#ifdef WITH_PCL
		const auto& point_xyz = point_cloud->points[i];
		float x = point_xyz.x; float y = point_xyz.y; float z = point_xyz.z;
#elif defined(WITH_CILANTRO)
		const auto& point_xyz = point_cloud->points.col(i);
		float x = point_xyz.x(); float y = point_xyz.y(); float z = point_xyz.z();
#endif
		setPointRGB(x, y, z, rgba.x, rgba.y, rgba.z, color_cloud, i,1.0f);

	}
	return color_cloud;
}

/* The index map query methods
 */
namespace surfelwarp {
namespace device {

__global__ void queryIndexMapFromPixelKernel(
		cudaTextureObject_t index_map,
		const DeviceArrayView<ushort4> pixel_array,
		unsigned* index_array
) {
	const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < pixel_array.Size()) {
		const auto x = pixel_array[idx].x;
		const auto y = pixel_array[idx].y;
		const auto index = tex2D<unsigned>(index_map, x, y);
		index_array[idx] = index;
	}
}


} // namespace device
} // namespace surfelwarp


void surfelwarp::queryIndexMapFromPixels(
		cudaTextureObject_t index_map,
		const DeviceArrayView<ushort4>& pixel_array,
		DeviceArray<unsigned>& index_array
) {
	//Simple sanity check
	SURFELWARP_CHECK_EQ(pixel_array.Size(), index_array.size());

	//Invoke the kernel
	dim3 blk(256);
	dim3 grid(pixel_array.Size(), blk.x);
	device::queryIndexMapFromPixelKernel << < grid, blk >> > (index_map, pixel_array, index_array);
}
