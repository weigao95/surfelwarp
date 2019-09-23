//
// Created by wei on 3/23/18.
//

#include "common/Stream.h"
#include "common/Serializer.h"
#include "common/BinaryFileStream.h"
#include "common/data_transfer.h"
#include "common/common_point_cloud_utils.h"
#include "visualization/Visualizer.h"
#include <common/ArrayView.h>

#ifdef WITH_PCL
#include <pcl/visualization/pcl_visualizer.h>
#elif defined(WITH_CILANTRO)

#include <cilantro/visualizer.hpp>
#include <cilantro/common_renderables.hpp>

#endif

/* The point cloud drawing methods
*/
void surfelwarp::Visualizer::DrawPointCloud(const PointCloud3f_Pointer &point_cloud) {
    const std::string window_title = "simple point cloud viewer";
#ifdef WITH_PCL
    pcl::visualization::PCLVisualizer viewer(window_title);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(point_cloud, 255, 255, 255);
    viewer.addPointCloud(point_cloud, "point cloud");
    viewer.addCoordinateSystem(2.0, "point cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "point cloud");
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
#elif defined(WITH_CILANTRO)
    cilantro::Visualizer viewer(window_title, "display");
    viewer.setClearColor(0.05f, 0.05f, 0.05f);
    viewer.addObject<cilantro::PointCloudRenderable>("point cloud", *point_cloud.get(),
                                                     cilantro::RenderingProperties()
                                                     .setPointColor(1.0, 1.0, 1.0)
                                                     .setPointSize(2.0));
    viewer.addObject<cilantro::CoordinateFrameRenderable>("axes", Eigen::Matrix4f::Identity(), 0.4f,
                                                          cilantro::RenderingProperties().setLineWidth(2.0));
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
#endif
}

void surfelwarp::Visualizer::DrawPointCloud(const surfelwarp::DeviceArray<float4> &point_cloud) {
    const auto h_point_cloud = downloadPointCloud(point_cloud);
    DrawPointCloud(h_point_cloud);
}

void surfelwarp::Visualizer::DrawPointCloud(const surfelwarp::DeviceArrayView<float4> &cloud) {
    DeviceArray<float4> cloud_array = DeviceArray<float4>((float4 *) cloud.RawPtr(), cloud.Size());
    DrawPointCloud(cloud_array);
}

void surfelwarp::Visualizer::DrawPointCloud(const DeviceArray2D<float4> &vertex_map) {
    const auto point_cloud = downloadPointCloud(vertex_map);
    DrawPointCloud(point_cloud);
}

void surfelwarp::Visualizer::DrawPointCloud(cudaTextureObject_t vertex_map) {
    const auto point_cloud = downloadPointCloud(vertex_map);
    DrawPointCloud(point_cloud);
}

void surfelwarp::Visualizer::DrawPointCloud(
        const surfelwarp::DeviceArray<surfelwarp::DepthSurfel> &surfel_array
) {
#ifdef WITH_PCL
    PointCloud3f_Pointer point_cloud;
    PointCloudNormal_Pointer normal_cloud;
    downloadPointNormalCloud(surfel_array, point_cloud, normal_cloud);
#elif defined(WITH_CILANTRO)
    PointCloud3f_Pointer point_cloud;
    downloadPointNormalCloud(surfel_array, point_cloud);
#endif
    DrawPointCloud(point_cloud);
}

void surfelwarp::Visualizer::SavePointCloud(const std::vector<float4> &point_vec, const std::string &path) {
    std::ofstream file_output;
    file_output.open(path);
    file_output << "OFF" << std::endl;
    file_output << point_vec.size() << " " << 0 << " " << 0 << std::endl;
    for (int node_iter = 0; node_iter < point_vec.size(); node_iter++) {
        file_output << point_vec[node_iter].x * 1000
                    << " " << point_vec[node_iter].y * 1000 << " "
                    << point_vec[node_iter].z * 1000
                    << std::endl;
    }
}

void surfelwarp::Visualizer::SavePointCloud(cudaTextureObject_t vertex_map, const std::string &path) {
    std::vector<float4> point_vec;
    downloadPointCloud(vertex_map, point_vec);
    std::ofstream file_output;
    file_output.open(path);
    file_output << "OFF" << std::endl;
    file_output << point_vec.size() << " " << 0 << " " << 0 << std::endl;
    for (int node_iter = 0; node_iter < point_vec.size(); node_iter++) {
        file_output << point_vec[node_iter].x * 1000
                    << " " << point_vec[node_iter].y * 1000 << " "
                    << point_vec[node_iter].z * 1000
                    << std::endl;
    }
}


void surfelwarp::Visualizer::SavePointCloud(const DeviceArrayView<float4> cloud, const std::string &path) {
    DeviceArray<float4> point_cloud((float4 *) cloud.RawPtr(), cloud.Size());
    std::vector<float4> point_vec;
    point_cloud.download(point_vec);
    SavePointCloud(point_vec, path);
}

/* The point cloud with normal
*/
void surfelwarp::Visualizer::DrawPointCloudWithNormal(
        const PointCloud3f_Pointer &point_cloud
#ifdef WITH_PCL
        ,const PointCloudNormal_Pointer & normal_cloud
#endif
) {
    const std::string window_title = "3D Viewer";
#ifdef WITH_PCL
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(point_cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(point_cloud, handler, "sample cloud");
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(point_cloud, normal_cloud, 30, 60.0f);
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
#elif defined(WITH_CILANTRO)
    cilantro::Visualizer viewer(window_title, "display");
    viewer.setClearColor(0.0f, 0.0f, 0.0f);
    viewer.addObject<cilantro::PointCloudRenderable>("point cloud", *point_cloud.get(),
                                                     cilantro::RenderingProperties()
                                                     .setPointColor(1.0, 1.0, 1.0)
                                                     .setPointSize(2.0)
                                                     .setDrawNormals(true)
                                                     .setNormalLength(60.0f)
                                                     .setLineDensityFraction(1.0f / 30.0f));
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
#endif
}

template<typename TPointInput, typename TNormalsInput>
void surfelwarp::Visualizer::DrawPointCloudWithNormals_Generic(TPointInput &points, TNormalsInput &normals) {
#ifdef WITH_PCL
    const auto point_cloud = downloadPointCloud(points);
    const auto normal_cloud = downloadNormalCloud(normals);
    DrawPointCloudWithNormal(point_cloud, normal_cloud);
#elif defined(WITH_CILANTRO)
    auto point_cloud = surfelwarp::downloadPointCloud(points);
    surfelwarp::downloadNormalCloud(normals, point_cloud);
    DrawPointCloudWithNormal(point_cloud);
#endif
}

void surfelwarp::Visualizer::DrawPointCloudWithNormal(
        const DeviceArray<float4> &vertex,
        const DeviceArray<float4> &normal
) {
    DrawPointCloudWithNormals_Generic(vertex, normal);

}

void surfelwarp::Visualizer::DrawPointCloudWithNormal(
        const DeviceArrayView<float4> &vertex_cloud,
        const DeviceArrayView<float4> &normal_cloud
) {
    SURFELWARP_CHECK(vertex_cloud.Size() == normal_cloud.Size());
    DeviceArray<float4> vertex_array((float4 *) vertex_cloud.RawPtr(), vertex_cloud.Size());
    DeviceArray<float4> normal_array((float4 *) normal_cloud.RawPtr(), normal_cloud.Size());
    DrawPointCloudWithNormal(vertex_array, normal_array);
}

void surfelwarp::Visualizer::DrawPointCloudWithNormal(
        const DeviceArray2D<float4> &vertex_map,
        const DeviceArray2D<float4> &normal_map
) {
    DrawPointCloudWithNormals_Generic(vertex_map, normal_map);
}

void surfelwarp::Visualizer::DrawPointCloudWithNormal(
        cudaTextureObject_t vertex_map,
        cudaTextureObject_t normal_map
) {
    DrawPointCloudWithNormals_Generic(vertex_map, normal_map);
}

void surfelwarp::Visualizer::DrawPointCloudWithNormal(
        const DeviceArray<DepthSurfel> &surfel_array
) {
#ifdef WITH_PCL
    PointCloud3f_Pointer point_cloud;
    PointCloudNormal_Pointer normal_cloud;
    downloadPointNormalCloud(surfel_array, point_cloud, normal_cloud);
    DrawPointCloudWithNormal(point_cloud, normal_cloud);
#elif defined(WITH_CILANTRO)
    PointCloud3f_Pointer point_cloud;
    downloadPointNormalCloud(surfel_array, point_cloud);
    DrawPointCloudWithNormal(point_cloud);
#endif
}

void surfelwarp::Visualizer::SavePointCloudWithNormal(cudaTextureObject_t vertex_map, cudaTextureObject_t normal_map) {
    //Download it
#ifdef WITH_PCL
    const auto point_cloud = downloadPointCloud(vertex_map);
    const auto normal_cloud = downloadNormalCloud(normal_map);
#elif defined(WITH_CILANTRO)
    auto point_cloud = downloadPointCloud(vertex_map);
    downloadNormalCloud(normal_map, point_cloud);
#endif

    //Construct the output stream
    BinaryFileStream output_fstream("pointnormal", BinaryFileStream::Mode::WriteOnly);

    //Prepare the test data
    std::vector<float4> save_vec;
    for (auto i = 0; i < point_cloud->points.size(); i++) {
#ifdef WITH_PCL
        save_vec.push_back(
                make_float4(point_cloud->points[i].x, point_cloud->points[i].y, point_cloud->points[i].z, 0));
        save_vec.push_back(make_float4(
                normal_cloud->points[i].normal_x,
                normal_cloud->points[i].normal_y,
                normal_cloud->points[i].normal_z,
                0));
#elif defined(WITH_CILANTRO)
        save_vec.push_back(
                make_float4(point_cloud->points(0, i),
                            point_cloud->points(1, i),
                            point_cloud->points(2, i), 0));
        save_vec.push_back(make_float4(
                point_cloud->normals(0, i),
                point_cloud->normals(1, i),
                point_cloud->normals(2, i), 0));
#endif

    }

    //Save it
    //PODVectorSerializeHandler<int>::Write(&output_fstream, save_vec);
    //SerializeHandler<std::vector<int>>::Write(&output_fstream, save_vec);
    //output_fstream.Write<std::vector<int>>(save_vec);
    output_fstream.SerializeWrite<std::vector<float4>>(save_vec);
}


/* The colored point cloud drawing method
 */
void surfelwarp::Visualizer::DrawColoredPointCloud(const PointCloud3fRGB_Pointer &point_cloud) {
    std::string window_title = "3D Viewer";
#ifdef WITH_PCL
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(point_cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
#elif defined(WITH_CILANTRO)
    cilantro::Visualizer viewer(window_title, "display");
    viewer.setClearColor(0.0f, 0.0f, 0.0f);
    viewer.addObject<cilantro::PointCloudRenderable>("point cloud", *point_cloud.get(),
                                                     cilantro::RenderingProperties().setPointSize(3.0));
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
#endif
}

void surfelwarp::Visualizer::SaveColoredPointCloud(const PointCloud3fRGB_Pointer &point_cloud,
                                                   const std::string &path) {
    std::ofstream file_output;
    file_output.open(path);
    const auto &points = point_cloud->points;

    file_output << "COFF" << std::endl;
    file_output << points.size() << " " << 0 << " " << 0 << std::endl;
    for (auto i = 0; i < points.size(); i++) {
#ifdef WITH_PCL
        const auto point = points[i];
        file_output << point.x
                    << " " << point.y
                    << " " << point.z
                    << " " << point.r / 255.f
                    << " " << point.g / 255.f
                    << " " << point.b / 255.f
                    << std::endl;
#elif defined(WITH_CILANTRO)
        file_output << points(0, i) << points(1, i) << points(2, i)
                    << point_cloud->colors(0, i) << point_cloud->colors(1, i) << point_cloud->colors(2, i)
                    << std::endl;
#endif
    }
    file_output.close();
}

void surfelwarp::Visualizer::DrawColoredPointCloud(
        const surfelwarp::DeviceArray<float4> &vertex,
        const surfelwarp::DeviceArray<float4> &color_time
) {
    auto point_cloud = downloadColoredPointCloud(vertex, color_time);
    DrawColoredPointCloud(point_cloud);
}

void surfelwarp::Visualizer::DrawColoredPointCloud(
        const surfelwarp::DeviceArrayView<float4> &vertex,
        const surfelwarp::DeviceArrayView<float4> &color_time
) {
    DeviceArray<float4> vertex_array((float4 *) vertex.RawPtr(), vertex.Size());
    DeviceArray<float4> color_time_array((float4 *) color_time.RawPtr(), color_time.Size());
    DrawColoredPointCloud(vertex_array, color_time_array);
}

void surfelwarp::Visualizer::DrawColoredPointCloud(cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map) {
    auto cloud = downloadColoredPointCloud(vertex_map, color_time_map, true);
    DrawColoredPointCloud(cloud);
}

void surfelwarp::Visualizer::SaveColoredPointCloud(
        cudaTextureObject_t vertex_map,
        cudaTextureObject_t color_time_map,
        const std::string &path
) {
    auto cloud = downloadColoredPointCloud(vertex_map, color_time_map, true);
    SaveColoredPointCloud(cloud, path);
}

/* The method to draw matched cloud pair
 */
void surfelwarp::Visualizer::DrawMatchedCloudPair(
        const PointCloud3f_Pointer &cloud_1,
        const PointCloud3f_Pointer &cloud_2
) {
    std::string window_title = "Matched Viewer";
#ifdef WITH_PCL
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(
            new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_1(cloud_1, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_1, handler_1, "cloud 1");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_2(cloud_2, 0, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_2, handler_2, "cloud 2");
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
#elif defined(WITH_CILANTRO)
    cilantro::Visualizer viewer(window_title, "display");
    viewer.setClearColor(0.0f, 0.0f, 0.0f);
    viewer.addObject<cilantro::PointCloudRenderable>("cloud 1", *cloud_1.get(),
                                                     cilantro::RenderingProperties().setPointColor(1.0f, 0.0f, 0.0f));
    viewer.addObject<cilantro::PointCloudRenderable>("cloud 2", *cloud_2.get(),
                                                     cilantro::RenderingProperties().setPointColor(0.0f, 1.0f, 1.0f));
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
#endif
}


void surfelwarp::Visualizer::DrawMatchedCloudPair(
        const PointCloud3f_Pointer &cloud_1,
        const PointCloud3f_Pointer &cloud_2,
        const Eigen::Matrix4f &from1To2
) {
    PointCloud3f_Pointer transformed_cloud_1 = transformPointCloud(cloud_1, from1To2);
    DrawMatchedCloudPair(transformed_cloud_1, cloud_2);
}

void surfelwarp::Visualizer::DrawMatchedCloudPair(
        cudaTextureObject_t cloud_1,
        const surfelwarp::DeviceArray<float4> &cloud_2,
        const surfelwarp::Matrix4f &from1To2
) {
    const auto h_cloud_1 = downloadPointCloud(cloud_1);
    const auto h_cloud_2 = downloadPointCloud(cloud_2);
    DrawMatchedCloudPair(h_cloud_1, h_cloud_2, from1To2);
}

void surfelwarp::Visualizer::DrawMatchedCloudPair(
        cudaTextureObject_t cloud_1,
        const DeviceArrayView<float4> &cloud_2,
        const Matrix4f &from1To2
) {
    DrawMatchedCloudPair(
            cloud_1,
            DeviceArray<float4>((float4 *) cloud_2.RawPtr(), cloud_2.Size()),
            from1To2
    );
}

void surfelwarp::Visualizer::DrawMatchedCloudPair(
        cudaTextureObject_t cloud_1,
        cudaTextureObject_t cloud_2,
        const surfelwarp::Matrix4f &from1To2
) {
    const auto h_cloud_1 = downloadPointCloud(cloud_1);
    const auto h_cloud_2 = downloadPointCloud(cloud_2);
    DrawMatchedCloudPair(h_cloud_1, h_cloud_2, from1To2);
}


void surfelwarp::Visualizer::SaveMatchedCloudPair(
        const PointCloud3f_Pointer &cloud_1,
        const PointCloud3f_Pointer &cloud_2,
        const std::string &cloud_1_name, const std::string &cloud_2_name
) {
    auto color_cloud_1 = addColorToPointCloud(cloud_1, make_uchar4(245, 0, 0, 255));
    auto color_cloud_2 = addColorToPointCloud(cloud_2, make_uchar4(200, 200, 200, 255));
    SaveColoredPointCloud(color_cloud_1, cloud_1_name);
    SaveColoredPointCloud(color_cloud_2, cloud_2_name);
}

void surfelwarp::Visualizer::SaveMatchedCloudPair(
        const PointCloud3f_Pointer &cloud_1,
        const PointCloud3f_Pointer &cloud_2,
        const Eigen::Matrix4f &from1To2,
        const std::string &cloud_1_name, const std::string &cloud_2_name
) {
    PointCloud3f_Pointer transformed_cloud_1 = transformPointCloud(cloud_1, from1To2);
    SaveMatchedCloudPair(transformed_cloud_1, cloud_2, cloud_1_name, cloud_2_name);
}


void surfelwarp::Visualizer::SaveMatchedCloudPair(
        cudaTextureObject_t cloud_1,
        const DeviceArray<float4> &cloud_2,
        const Eigen::Matrix4f &from1To2,
        const std::string &cloud_1_name, const std::string &cloud_2_name
) {
    const auto h_cloud_1 = downloadPointCloud(cloud_1);
    const auto h_cloud_2 = downloadPointCloud(cloud_2);
    SaveMatchedCloudPair(
            h_cloud_1,
            h_cloud_2,
            from1To2,
            cloud_1_name, cloud_2_name
    );
}


void surfelwarp::Visualizer::SaveMatchedCloudPair(
        cudaTextureObject_t cloud_1,
        const DeviceArrayView<float4> &cloud_2,
        const Eigen::Matrix4f &from1To2,
        const std::string &cloud_1_name, const std::string &cloud_2_name
) {
    SaveMatchedCloudPair(
            cloud_1,
            DeviceArray<float4>((float4 *) cloud_2.RawPtr(), cloud_2.Size()),
            from1To2,
            cloud_1_name, cloud_2_name
    );
}


/* The method to draw mached color point cloud
 */
void surfelwarp::Visualizer::DrawMatchedRGBCloudPair(const PointCloud3fRGB_Pointer &cloud_1,
                                                     const PointCloud3fRGB_Pointer &cloud_2
) {
    std::string window_title = "3D Viewer";
#ifdef WITH_PCL
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(window_title));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_1(cloud_1);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud_1, handler_1, "cloud_1");
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_2(cloud_2);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud_2, handler_2, "cloud_2");

    //viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_1");
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
#else
    cilantro::Visualizer viewer(window_title, "display");
    viewer.setClearColor(0.0f, 0.0f, 0.0f);
    viewer.addObject<cilantro::PointCloudRenderable>("cloud 1", *cloud_1.get(),
                                                     cilantro::RenderingProperties().setPointSize(3.0));
    viewer.addObject<cilantro::PointCloudRenderable>("cloud 2", *cloud_1.get(),
                                                     cilantro::RenderingProperties().setPointSize(3.0));
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
#endif
}


void surfelwarp::Visualizer::DrawMatchedRGBCloudPair(
        const PointCloud3fRGB_Pointer &cloud_1,
        const PointCloud3fRGB_Pointer &cloud_2,
        const Eigen::Matrix4f &from1To2
) {
    PointCloud3fRGB_Pointer transformed_cloud_1 = transformPointCloudRGB(cloud_1, from1To2);

    //Hand in to drawer
	DrawMatchedRGBCloudPair(transformed_cloud_1, cloud_2);
}

void surfelwarp::Visualizer::DrawMatchedCloudPair(
        cudaTextureObject_t vertex_map, cudaTextureObject_t color_time_map,
        const DeviceArrayView<float4> &surfel_array,
        const DeviceArrayView<float4> &color_time_array,
        const Eigen::Matrix4f &camera2world
) {
    auto cloud_1 = downloadColoredPointCloud(vertex_map, color_time_map, true);
    auto cloud_2 = downloadColoredPointCloud(
            DeviceArray<float4>((float4 *) surfel_array.RawPtr(), surfel_array.Size()),
            DeviceArray<float4>((float4 *) color_time_array.RawPtr(), color_time_array.Size())
    );
	DrawMatchedRGBCloudPair(cloud_1, cloud_2, camera2world);
}


/* The method to draw fused surfel cloud
 */
void surfelwarp::Visualizer::DrawFusedSurfelCloud(
        DeviceArrayView<float4> surfel_vertex,
        DeviceArrayView<unsigned> fused_indicator
) {
    SURFELWARP_CHECK_EQ(surfel_vertex.Size(), fused_indicator.Size());

    //Construct the host cloud
    PointCloud3f_Pointer fused_cloud(new PointCloud3f);
    PointCloud3f_Pointer unfused_cloud(new PointCloud3f);

    //Download it
    separateDownloadPointCloud(surfel_vertex, fused_indicator, fused_cloud, unfused_cloud);

    //Ok draw it
    DrawMatchedCloudPair(fused_cloud, unfused_cloud, Eigen::Matrix4f::Identity());
}


void surfelwarp::Visualizer::DrawFusedSurfelCloud(
        surfelwarp::DeviceArrayView<float4> surfel_vertex,
        unsigned num_remaining_surfels
) {
    SURFELWARP_CHECK(surfel_vertex.Size() >= num_remaining_surfels);

    //Construct the host cloud
    PointCloud3f_Pointer remaining_cloud(new PointCloud3f);
    PointCloud3f_Pointer appended_cloud(new PointCloud3f);

    //Download it
    separateDownloadPointCloud(surfel_vertex, num_remaining_surfels, remaining_cloud, appended_cloud);

    //Ok draw it
    DrawMatchedCloudPair(remaining_cloud, appended_cloud);
}

void surfelwarp::Visualizer::DrawFusedAppendedSurfelCloud(
        surfelwarp::DeviceArrayView<float4> surfel_vertex,
        surfelwarp::DeviceArrayView<unsigned int> fused_indicator,
        cudaTextureObject_t depth_vertex_map,
        surfelwarp::DeviceArrayView<unsigned int> append_indicator,
        const surfelwarp::Matrix4f &world2camera
) {
    SURFELWARP_CHECK_EQ(surfel_vertex.Size(), fused_indicator.Size());

    //Construct the host cloud
    PointCloud3f_Pointer fused_cloud(new PointCloud3f);
    PointCloud3f_Pointer unfused_cloud(new PointCloud3f);

    //Download it
    separateDownloadPointCloud(surfel_vertex, fused_indicator, fused_cloud, unfused_cloud);
    auto h_append_surfels = downloadPointCloud(depth_vertex_map, append_indicator);

    //Draw it
    DrawMatchedCloudPair(fused_cloud, h_append_surfels, world2camera);
}

void surfelwarp::Visualizer::DrawAppendedSurfelCloud(
        DeviceArrayView<float4> surfel_vertex,
        cudaTextureObject_t depth_vertex_map,
        DeviceArrayView<unsigned int> append_indicator,
        const surfelwarp::Matrix4f &world2camera
) {
    auto h_surfels = downloadPointCloud(DeviceArray<float4>((float4 *) surfel_vertex.RawPtr(), surfel_vertex.Size()));
    auto h_append_surfels = downloadPointCloud(depth_vertex_map, append_indicator);
    DrawMatchedCloudPair(h_surfels, h_append_surfels, world2camera);
}


void surfelwarp::Visualizer::DrawAppendedSurfelCloud(
        DeviceArrayView<float4> surfel_vertex,
        cudaTextureObject_t depth_vertex_map,
        DeviceArrayView<ushort2> append_pixel,
        const surfelwarp::Matrix4f &world2camera
) {
    auto h_surfels = downloadPointCloud(DeviceArray<float4>((float4 *) surfel_vertex.RawPtr(), surfel_vertex.Size()));
    auto h_append_surfels = downloadPointCloud(depth_vertex_map, append_pixel);
    DrawMatchedCloudPair(h_surfels, h_append_surfels, world2camera);
}














