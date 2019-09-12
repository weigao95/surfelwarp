//
// Created by wei on 2/20/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/surfel_types.h"
#include "common/ArrayView.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include "common/point_cloud_typedefs.h"

namespace surfelwarp {


	/**
     * \brief This class is for DEBUG visualization. Any methods 
     *        in this class should NOT be used in real-time code.
     */
    class Visualizer {
    public:
        using Ptr = std::shared_ptr<Visualizer>;
	    
		SURFELWARP_DEFAULT_CONSTRUCT_DESTRUCT(Visualizer);
		SURFELWARP_NO_COPY_ASSIGN_MOVE(Visualizer);

		/* The depth image drawing methods
		 */
		static void DrawDepthImage(const cv::Mat& depth_img);
        static void SaveDepthImage(const cv::Mat& depth_img, const std::string& path);
		static void DrawDepthImage(const DeviceArray2D<unsigned short>& depth_img);
		static void SaveDepthImage(const DeviceArray2D<unsigned short>& depth_img, const std::string& path);
    	static void DrawDepthImage(cudaTextureObject_t depth_img);
		static void SaveDepthImage(cudaTextureObject_t depth_img, const std::string& path);

		/* The color image drawing methods
		*/
        static void DrawRGBImage(const cv::Mat& rgb_img);
        static void SaveRGBImage(const cv::Mat& rgb_img, const std::string& path);
		static void DrawRGBImage(const DeviceArray<uchar3>& rgb_img, const int rows, const int cols);
		static void SaveRGBImage(const DeviceArray<uchar3>& rgb_img, const int rows, const int cols, const std::string& path);
    	static void DrawNormalizeRGBImage(cudaTextureObject_t rgb_img);
		static void SaveNormalizeRGBImage(cudaTextureObject_t rgb_img, const std::string& path);
		static void DrawColorTimeMap(cudaTextureObject_t color_time_map);
	    static void DrawNormalMap(cudaTextureObject_t normal_map);
	    
	    /* The gray scale image drawing for filtered
	     */
	    static void DrawGrayScaleImage(const cv::Mat& gray_scale_img);
	    static void SaveGrayScaleImage(const cv::Mat& gray_scale_img, const std::string& path);
	    static void DrawGrayScaleImage(cudaTextureObject_t gray_scale_img, float scale = 1.0f);
	    static void SaveGrayScaleImage(cudaTextureObject_t gray_scale_img, const std::string& path, float scale = 1.0f);

	    
		/* The segmentation mask drawing methods
		*/
		static void MarkSegmentationMask(
			const std::vector<unsigned char>& mask, 
			cv::Mat& rgb_img,
			const unsigned sample_rate = 2
		);
		static void DrawSegmentMask(
			const std::vector<unsigned char>& mask, 
			cv::Mat& rgb_img, 
			const unsigned sample_rate = 2
		);
		static void SaveSegmentMask(
			const std::vector<unsigned char>& mask, 
			cv::Mat& rgb_img,
			const std::string& path, 
			const unsigned sample_rate = 2
		);
		static void DrawSegmentMask(
			cudaTextureObject_t mask, 
			cudaTextureObject_t normalized_rgb_img,
			const unsigned sample_rate = 2
		);
		static void SaveSegmentMask(
			cudaTextureObject_t mask, 
			cudaTextureObject_t normalized_rgb_img, 
			const std::string& path, 
			const unsigned sample_rate = 2
		);
	    static void SaveRawSegmentMask(
		    cudaTextureObject_t mask,
		    const std::string& path
	    );
	    static void DrawRawSegmentMask(
		    cudaTextureObject_t mask
	    );


		/* The binary meanfield drawing methods
		*/
		static void DrawBinaryMeanfield(cudaTextureObject_t meanfield_q);
		static void SaveBinaryMeanfield(cudaTextureObject_t meanfield_q, const std::string& path);
	    
	    
	    /* Visualize the valid geometry maps as binary mask
	     */
	    static void DrawValidIndexMap(cudaTextureObject_t index_map, int validity_halfsize);
	    static void SaveValidIndexMap(cudaTextureObject_t index_map, int validity_halfsize, const std::string& path);
	    
	    static cv::Mat GetValidityMapCV(cudaTextureObject_t index_map, int validity_halfsize);
	    //Mark the validity of each index map pixel and save them to flatten indicator
	    //Assume pre-allcoated indicator
	    static void MarkValidIndexMapValue(
		    cudaTextureObject_t index_map,
		    int validity_halfsize,
		    DeviceArray<unsigned char> flatten_validity_indicator
	    );
	    
	    
	    
	    /* The correspondence
	     */
	    static void DrawImagePairCorrespondence(
		    cudaTextureObject_t rgb_0, cudaTextureObject_t rgb_1,
		    const DeviceArray<ushort4>& correspondence
	    );


		/* The point cloud drawing methods
		*/
		static void DrawPointCloud(const PointCloud3f_Pointer& point_cloud);
	    static void DrawPointCloud(const DeviceArray<float4>& point_cloud);
	    static void DrawPointCloud(const DeviceArrayView<float4>& point_cloud);
		static void DrawPointCloud(const DeviceArray2D<float4>& vertex_map);
	    static void DrawPointCloud(const DeviceArray<DepthSurfel>& surfel_array);
		static void DrawPointCloud(cudaTextureObject_t vertex_map);
	    static void SavePointCloud(const std::vector<float4>& point_cloud, const std::string& path);
		static void SavePointCloud(cudaTextureObject_t veretx_map, const std::string& path);
	    static void SavePointCloud(const DeviceArrayView<float4> point_cloud, const std::string& path);


		/* The point cloud with normal
		 */
		static void DrawPointCloudWithNormal(
			const PointCloud3f_Pointer& point_cloud
#ifdef WITH_PCL
			,const PointCloudNormal_Pointer& normal_cloud
#endif
		);
	    static void DrawPointCloudWithNormal(
		    const DeviceArray<float4>& vertex_cloud,
		    const DeviceArray<float4>& normal_cloud
	    );
	    static void DrawPointCloudWithNormal(
		    const DeviceArrayView<float4>& vertex_cloud,
		    const DeviceArrayView<float4>& normal_cloud
	    );
		static void DrawPointCloudWithNormal(
			const DeviceArray2D<float4>& vertex_map,
			const DeviceArray2D<float4>& normal_map
		);
		static void DrawPointCloudWithNormal(cudaTextureObject_t vertex_map, cudaTextureObject_t normal_map);
	    static void DrawPointCloudWithNormal(const DeviceArray<DepthSurfel>& surfel_array);
	    static void SavePointCloudWithNormal(cudaTextureObject_t vertex_map, cudaTextureObject_t normal_map);
	    
	    
	    /* The colored point cloud
	     */
	    static void DrawColoredPointCloud(const PointCloud3fRGB_Pointer& point_cloud);
	    static void SaveColoredPointCloud(const PointCloud3fRGB_Pointer& point_cloud, const std::string& path);
	    static void DrawColoredPointCloud(const DeviceArray<float4>& vertex, const DeviceArray<float4>& color_time);
	    static void DrawColoredPointCloud(const DeviceArrayView<float4>& vertex, const DeviceArrayView<float4>& color_time);
	    static void DrawColoredPointCloud(cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map);
	    static void SaveColoredPointCloud(cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map, const std::string& path);
	    
	    /* The matched point cloud
	     */
	    static void DrawMatchedCloudPair(const PointCloud3f_Pointer& cloud_1,
	                                     const PointCloud3f_Pointer& cloud_2);
	    static void DrawMatchedCloudPair(const PointCloud3f_Pointer& cloud_1,
	                                     const PointCloud3f_Pointer& cloud_2,
	                                     const Eigen::Matrix4f & from1To2);
	    static void DrawMatchedCloudPair(
			cudaTextureObject_t cloud_1,
	        const DeviceArray<float4>& cloud_2,
	        const Matrix4f& from1To2);
		static void DrawMatchedCloudPair(
			cudaTextureObject_t cloud_1,
			const DeviceArrayView<float4>& cloud_2,
			const Matrix4f& from1To2);
	    static void DrawMatchedCloudPair(cudaTextureObject_t cloud_1,
	                                     cudaTextureObject_t cloud_2,
	                                     const Matrix4f& from1To2);
	
	
	    static void SaveMatchedCloudPair(
		    const PointCloud3f_Pointer& cloud_1,
		    const PointCloud3f_Pointer& cloud_2,
		    const std::string& cloud_1_name, const std::string& cloud_2_name
	    );
	    static void SaveMatchedCloudPair(
		    const PointCloud3f_Pointer & cloud_1,
		    const PointCloud3f_Pointer & cloud_2,
		    const Eigen::Matrix4f & from1To2,
		    const std::string& cloud_1_name, const std::string& cloud_2_name
	    );
	    static void SaveMatchedCloudPair(
		    cudaTextureObject_t cloud_1,
		    const DeviceArray<float4>& cloud_2,
		    const Matrix4f& from1To2,
		    const std::string& cloud_1_name, const std::string& cloud_2_name
	    );
	    static void SaveMatchedCloudPair(
		    cudaTextureObject_t cloud_1,
		    const DeviceArrayView<float4>& cloud_2,
		    const Matrix4f& from1To2,
		    const std::string& cloud_1_name, const std::string& cloud_2_name
	    );
	
	
	    /* The method to draw matched color-point cloud
	     */
	    static void DrawMatchedRGBCloudPair(const PointCloud3fRGB_Pointer& cloud_1,
	                                        const PointCloud3fRGB_Pointer& cloud_2);
	    static void DrawMatchedRGBCloudPair(const PointCloud3fRGB_Pointer& cloud_1,
	                                        const PointCloud3fRGB_Pointer& cloud_2,
	                                        const Eigen::Matrix4f& from1To2);
	    static void DrawMatchedCloudPair(
		    cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map,
		    const DeviceArrayView<float4>& surfel_array, const DeviceArrayView<float4>& color_time_array,
		    const Eigen::Matrix4f& camera2world
	    );



		/* The method to draw fused point cloud
		 */
		static void DrawFusedSurfelCloud(
			DeviceArrayView<float4> surfel_vertex,
			DeviceArrayView<unsigned> fused_indicator
		);
	    static void DrawFusedSurfelCloud(
		    DeviceArrayView<float4> surfel_vertex,
		    unsigned num_remaining_surfels
	    );
	    
	    static void DrawFusedAppendedSurfelCloud(
		    DeviceArrayView<float4> surfel_vertex,
		    DeviceArrayView<unsigned> fused_indicator,
		    cudaTextureObject_t depth_vertex_map,
		    DeviceArrayView<unsigned> append_indicator,
		    const Matrix4f& world2camera
	    );
	    
	    static void DrawAppendedSurfelCloud(
		    DeviceArrayView<float4> surfel_vertex,
		    cudaTextureObject_t depth_vertex_map,
		    DeviceArrayView<unsigned> append_indicator,
		    const Matrix4f& world2camera
	    );
	    static void DrawAppendedSurfelCloud(
		    DeviceArrayView<float4> surfel_vertex,
		    cudaTextureObject_t depth_vertex_map,
		    DeviceArrayView<ushort2> append_pixel,
		    const Matrix4f& world2camera
	    );
    private:
        template<typename TPointInput, typename TNormalsInput>
        static void DrawPointCloudWithNormals_Generic(TPointInput& points, TNormalsInput& normals);
    };
}
