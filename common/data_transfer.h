#pragma once

#include "common/common_types.h"
#include "common/surfel_types.h"
#include "common/ArrayView.h"
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


namespace surfelwarp
{
	/* Download the image from GPU memory to CPU memory
	 */
	cv::Mat downloadDepthImage(const DeviceArray2D<unsigned short>& image_gpu);
	cv::Mat downloadDepthImage(cudaTextureObject_t image_gpu);
	cv::Mat downloadRGBImage(
		const DeviceArray<uchar3>& image_gpu,
		const unsigned rows, const unsigned cols
	);

	//The rgb texture is in float4
	cv::Mat downloadNormalizeRGBImage(const DeviceArray2D<float4>& rgb_img);
	cv::Mat downloadNormalizeRGBImage(cudaTextureObject_t rgb_img);
	cv::Mat rgbImageFromColorTimeMap(cudaTextureObject_t color_time_map);
	cv::Mat normalMapForVisualize(cudaTextureObject_t normal_map);

	//The segmentation mask texture
	void downloadSegmentationMask(cudaTextureObject_t mask, std::vector<unsigned char>& h_mask);
	cv::Mat downloadRawSegmentationMask(cudaTextureObject_t mask); //uchar texture
	
	//The gray scale image
	void downloadGrayScaleImage(cudaTextureObject_t image, cv::Mat& h_image, float scale = 1.0f);

	//The binary meanfield map, the texture contains the
	//mean field probability of the positive label
	void downloadTransferBinaryMeanfield(cudaTextureObject_t meanfield_q, cv::Mat& h_meanfield_uchar);


	/* The point cloud download functions
	 */
	pcl::PointCloud<pcl::PointXYZ>::Ptr downloadPointCloud(const DeviceArray<float4>& vertex);
	pcl::PointCloud<pcl::PointXYZ>::Ptr downloadPointCloud(const DeviceArray2D<float4>& vertex_map);
	pcl::PointCloud<pcl::PointXYZ>::Ptr downloadPointCloud(const DeviceArray2D<float4>& vertex_map, DeviceArrayView<unsigned> indicator);
	pcl::PointCloud<pcl::PointXYZ>::Ptr downloadPointCloud(const DeviceArray2D<float4>& vertex_map, DeviceArrayView<ushort2> pixel);
	void downloadPointCloud(const DeviceArray2D<float4>& vertex_map, std::vector<float4>& point_cloud);
	pcl::PointCloud<pcl::PointXYZ>::Ptr downloadPointCloud(cudaTextureObject_t vertex_map);
	pcl::PointCloud<pcl::PointXYZ>::Ptr downloadPointCloud(cudaTextureObject_t vertex_map, DeviceArrayView<unsigned> indicator);
	pcl::PointCloud<pcl::PointXYZ>::Ptr downloadPointCloud(cudaTextureObject_t vertex_map, DeviceArrayView<ushort2> pixel);
	void downloadPointCloud(cudaTextureObject_t vertex_map, std::vector<float4>& point_cloud);
	void downloadPointNormalCloud(
		const DeviceArray<DepthSurfel>& surfel_array,
		pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud,
		pcl::PointCloud<pcl::Normal>::Ptr& normal_cloud,
		const float point_scale = 1000.0f
	);
	
	//Download it with indicator
	void separateDownloadPointCloud(
		const DeviceArrayView<float4>& point_cloud,
		const DeviceArrayView<unsigned>& indicator,
		pcl::PointCloud<pcl::PointXYZ>& fused_cloud,
		pcl::PointCloud<pcl::PointXYZ>& unfused_cloud
	);
	void separateDownloadPointCloud(
		const DeviceArrayView<float4>& point_cloud,
		unsigned num_remaining_surfels,
		pcl::PointCloud<pcl::PointXYZ>& remaining_cloud,
		pcl::PointCloud<pcl::PointXYZ>& appended_cloud
	);

	/* The normal cloud download functions
	*/
	pcl::PointCloud<pcl::Normal>::Ptr downloadNormalCloud(const DeviceArray<float4>& normal_cloud);
	pcl::PointCloud<pcl::Normal>::Ptr downloadNormalCloud(const DeviceArray2D<float4>& normal_map);
	pcl::PointCloud<pcl::Normal>::Ptr downloadNormalCloud(cudaTextureObject_t normal_map);
	
	
	/* The colored point cloud download function
	 */
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr downloadColoredPointCloud(
		const DeviceArray<float4>& vertex_confid,
		const DeviceArray<float4>& color_time
	);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr downloadColoredPointCloud(
		cudaTextureObject_t vertex_map,
		cudaTextureObject_t color_time_map,
		bool flip_color = false
	);
	
	
	/* Colorize the point cloud
	 */
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr addColorToPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, uchar4 rgba);
	

	/* Query the index map
	 */
	void queryIndexMapFromPixels(cudaTextureObject_t index_map, const DeviceArrayView<ushort4>& pixel_array, DeviceArray<unsigned>& knn_array);

	/* Transfer the memory from texture to GPU memory.
	 * Assume ALLOCATED device memory.
	 */
	template<typename T>
	void textureToMap2D(
		cudaTextureObject_t texture, 
		DeviceArray2D<T>& map, 
		cudaStream_t stream = 0
	);
}

#if defined(__CUDACC__)
#include "common/data_transfer.cuh"
#endif