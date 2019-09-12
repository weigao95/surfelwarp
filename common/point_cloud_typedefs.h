#pragma once

#ifdef WITH_PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
typedef pcl::PointCloud<pcl::PointXYZ>    PointCloud3f;
typedef pcl::PointCloud<pcl::Normal>      PointCloudNormal;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud3fRGB;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr    PointCloud3f_Pointer;
typedef pcl::PointCloud<pcl::Normal>::Ptr      PointCloudNormal_Pointer;
typedef pcl::PointCloud<pcl::PointXYZRGB>::Ptr PointCloud3fRGB_Pointer;
#elif defined(WITH_CILANTRO)
#include <cilantro/point_cloud.hpp>
typedef cilantro::PointCloud3f PointCloud3f;
typedef cilantro::PointCloud3f PointCloudNormal;
typedef cilantro::PointCloud3f PointCloud3fRGB;
typedef std::shared_ptr<cilantro::PointCloud3f> PointCloud3f_Pointer;
typedef std::shared_ptr<cilantro::PointCloud3f> PointCloudNormal_Pointer;
typedef std::shared_ptr<cilantro::PointCloud3f> PointCloud3fRGB_Pointer;
#endif
