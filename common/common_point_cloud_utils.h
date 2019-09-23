#pragma once

#include "common/point_cloud_typedefs.h"


inline
void setPoint(float x, float y, float z, PointCloud3f_Pointer& point_cloud, int index, float scale = 1000.0f) {
#ifdef WITH_PCL
    pcl::PointXYZ point;
	point.x = x * scale;
	point.y = y * scale;
	point.z = z * scale;
	point_cloud->points.push_back(point);
#elif defined(WITH_CILANTRO)
    point_cloud->points.col(index) = Eigen::Vector3f(x * scale, y * scale, z * scale);
#endif
}

inline
void setPointRGB(float x, float y, float z, unsigned char r, unsigned char g, unsigned char b, PointCloud3fRGB_Pointer& point_cloud, int index,
                 float scale = 1000.0f) {
#ifdef WITH_PCL
    pcl::PointXYZRGB point;
	point.x = x * scale;
	point.y = y * scale;
	point.z = z * scale;
	point.r = r;
	point.g = g;
	point.b = b;
	point_cloud->points.push_back(point);
#elif defined(WITH_CILANTRO)
    point_cloud->points.col(index) = Eigen::Vector3f(x * scale, y * scale, z * scale);
    point_cloud->colors.col(index) = Eigen::Vector3f(static_cast<float>(r) / 255.f, static_cast<float>(g) / 255.f,
                                                     static_cast<float>(b) / 255.f);
#endif
}

inline
void setNormal(float x, float y, float z, PointCloudNormal_Pointer& normal_cloud, int index) {
#ifdef WITH_PCL
    pcl::Normal normal;
	normal.normal_x = x;
	normal.normal_y = y;
	normal.normal_z = z;
	normal_cloud->points.push_back(normal);
#elif defined(WITH_CILANTRO)
    normal_cloud->normals.col(index) = Eigen::Vector3f(x, y, z);
#endif
}

inline
void setPointCloudSize(PointCloud3f_Pointer& point_cloud, int size) {
#ifdef WITH_CILANTRO
    point_cloud->points.resize(PointCloud3f::Dimension, size);
#endif
}

inline
void setPointCloudRGBSize(PointCloud3fRGB_Pointer& point_cloud, int size) {
#ifdef WITH_CILANTRO
    point_cloud->points.resize(PointCloud3f::Dimension, size);
    point_cloud->colors.resize(3, size);
#endif
}

inline
void setNormalCloudSize(PointCloudNormal_Pointer& point_cloud, int size) {
#ifdef WITH_CILANTRO
    point_cloud->normals.resize(PointCloudNormal::Dimension, size);
#endif
}

inline
PointCloud3f_Pointer transformPointCloud(const PointCloud3f_Pointer& input, Eigen::Matrix4f transformation){
    PointCloud3f_Pointer transformed_cloud(new PointCloud3f());
    setPointCloudSize(transformed_cloud, input->size());
    for (auto iPoint = 0; iPoint < input->points.size(); iPoint++) {
#ifdef WITH_PCL
        auto &point = input->points[iPoint];
        float x = point.x * 0.001;
        float y = point.y * 0.001;
        float z = point.z * 0.001;
#elif defined(WITH_CILANTRO)
        float x = input->points(0, iPoint) * 0.001;
        float y = input->points(1, iPoint) * 0.001;
        float z = input->points(2, iPoint) * 0.001;
#endif
        Eigen::Vector4f transed_point4(x, y, z, 1.0);
        transed_point4 = transformation * transed_point4;
#ifdef WITH_PCL
        pcl::PointXYZ transed_point;
        transed_point.x = transed_point4(0) * 1000;
        transed_point.y = transed_point4(1) * 1000;
        transed_point.z = transed_point4(2) * 1000;
        transformed_cloud->points.push_back(transed_point);
#elif defined(WITH_CILANTRO)
        transformed_cloud->points.col(iPoint) = transed_point4.head<3>();
#endif
    }
    return transformed_cloud;
}

inline
PointCloud3fRGB_Pointer transformPointCloudRGB(const PointCloud3fRGB_Pointer& input, Eigen::Matrix4f transformation){
    PointCloud3fRGB_Pointer transformed_cloud(new PointCloud3fRGB());
    setPointCloudRGBSize(transformed_cloud, input->size());
    for (auto iPoint = 0; iPoint < input->points.size(); iPoint++) {
#ifdef WITH_PCL
        auto &point = input->points[iPoint];
        float x = point.x * 0.001;
        float y = point.y * 0.001;
        float z = point.z * 0.001;
#elif defined(WITH_CILANTRO)
        float x = input->points(0, iPoint) * 0.001;
        float y = input->points(1, iPoint) * 0.001;
        float z = input->points(2, iPoint) * 0.001;
#endif
        Eigen::Vector4f transed_point4(x, y, z, 1.0);
        transed_point4 = transformation * transed_point4;
#ifdef WITH_PCL
        pcl::PointXYZRGB transed_point;
        transed_point.x = transed_point4(0) * 1000;
        transed_point.y = transed_point4(1) * 1000;
        transed_point.z = transed_point4(2) * 1000;
        transed_point.r = point.r;
        transed_point.g = point.g;
        transed_point.b = point.b;
        transformed_cloud->points.push_back(transed_point);
#elif defined(WITH_CILANTRO)
        transformed_cloud->points.col(iPoint) = transed_point4.head<3>();
        transformed_cloud->colors.col(iPoint) = static_cast<Eigen::Vector3f>(input->colors.col(iPoint));
#endif
    }
    return transformed_cloud;
}