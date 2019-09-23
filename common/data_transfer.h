#pragma once

#include "common/common_types.h"
#include "common/surfel_types.h"
#include "common/ArrayView.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include "common/point_cloud_typedefs.h"


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
	PointCloud3f_Pointer downloadPointCloud(const DeviceArray<float4>& vertex);
	PointCloud3f_Pointer downloadPointCloud(const DeviceArray2D<float4>& vertex_map);
	PointCloud3f_Pointer downloadPointCloud(const DeviceArray2D<float4>& vertex_map, DeviceArrayView<unsigned> indicator);
	PointCloud3f_Pointer downloadPointCloud(const DeviceArray2D<float4>& vertex_map, DeviceArrayView<ushort2> pixel);
	void downloadPointCloud(const DeviceArray2D<float4>& vertex_map, std::vector<float4>& point_cloud);
	PointCloud3f_Pointer downloadPointCloud(cudaTextureObject_t vertex_map);
	PointCloud3f_Pointer downloadPointCloud(cudaTextureObject_t vertex_map, DeviceArrayView<unsigned> indicator);
	PointCloud3f_Pointer downloadPointCloud(cudaTextureObject_t vertex_map, DeviceArrayView<ushort2> pixel);
	void downloadPointCloud(cudaTextureObject_t vertex_map, std::vector<float4>& point_cloud);


	void downloadPointNormalCloud(
		const DeviceArray<DepthSurfel>& surfel_array,
		PointCloud3f_Pointer& point_cloud,
#ifdef WITH_PCL
		PointCloudNormal_Pointer& normal_cloud,
#endif
		const float point_scale = 1000.0f
	);

	//Download it with indicator
	void separateDownloadPointCloud(
		const DeviceArrayView<float4>& point_cloud,
		const DeviceArrayView<unsigned>& indicator,
		PointCloud3f_Pointer& fused_cloud,
		PointCloud3f_Pointer& unfused_cloud
	);
	void separateDownloadPointCloud(
		const DeviceArrayView<float4>& point_cloud,
		unsigned num_remaining_surfels,
		PointCloud3f_Pointer& remaining_cloud,
		PointCloud3f_Pointer& appended_cloud
	);

	/* The normal cloud download functions
	*/
#ifdef WITH_PCL
	PointCloudNormal_Pointer downloadNormalCloud(const DeviceArray<float4>& normal_cloud);
	PointCloudNormal_Pointer downloadNormalCloud(const DeviceArray2D<float4>& normal_map);
	PointCloudNormal_Pointer downloadNormalCloud(cudaTextureObject_t normal_map);
#elif defined(WITH_CILANTRO)
	void downloadNormalCloud(const DeviceArray<float4>& normal_cloud, PointCloudNormal_Pointer& point_cloud);
	void downloadNormalCloud(const DeviceArray2D<float4>& normal_map, PointCloudNormal_Pointer& point_cloud);
	void downloadNormalCloud(cudaTextureObject_t normal_map, PointCloudNormal_Pointer& point_cloud);
#endif


	/* The colored point cloud download function
	 */
	PointCloud3fRGB_Pointer downloadColoredPointCloud(
		const DeviceArray<float4>& vertex_confid,
		const DeviceArray<float4>& color_time
	);
	PointCloud3fRGB_Pointer downloadColoredPointCloud(
		cudaTextureObject_t vertex_map,
		cudaTextureObject_t color_time_map,
		bool flip_color = false
	);


	/* Colorize the point cloud
	 */
	PointCloud3fRGB_Pointer addColorToPointCloud(const PointCloud3f_Pointer& point_cloud, uchar4 rgba);


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