#pragma once
#include <opencv2/opencv.hpp>

#if CV_VERSION_MAJOR >= 4
const int CV_ANYCOLOR = cv::IMREAD_ANYCOLOR;
const int CV_ANYDEPTH = cv::IMREAD_ANYDEPTH;
#else
const int CV_ANYCOLOR = CV_LOAD_IMAGE_ANYCOLOR;
const int CV_ANYDEPTH = CV_LOAD_IMAGE_ANYDEPTH;
#endif