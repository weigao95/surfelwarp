#include "common/common_types.h"
#include "common/common_utils.h"
#include "common/ConfigParser.h"
#include "common/sanity_check.h"
#include "common/CameraObservation.h"
#include "visualization/Visualizer.h"
#include "imgproc/frameio/FetchInterface.h"
#include "imgproc/frameio/GenericFileFetch.h"
// #include "imgproc/frameio/AzureKinectDKFetch.h"
#include "imgproc/frameio/VolumeDeformFileFetch.h"
#include "imgproc/frameio/RealsenseFetch.h"
#include "imgproc/ImageProcessor.h"

#include <thread>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace surfelwarp;

int holes_fill = 1;
int filter_magnitude = 3;
int smooth_alpha = 50; // 滑动条范围为0-100，实际使用时除以100
int smooth_delta = 15;
int holes_fill_filter = 0;

void on_trackbar(int, void *) {}

// 测试滤波器
void testFilter()
{
	rs2::pipeline pipe;
	rs2::config cfg;

	// 启用深度和颜色流
	cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
	cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

	// 启动管道
	rs2::pipeline_profile profile = pipe.start(cfg);

	// 初始化空间滤波器
	rs2::spatial_filter m_spatial_filter;
	rs2::hole_filling_filter m_hole_fill_filter;
	rs2::disparity_transform m_depth_to_disparity(true);
	rs2::disparity_transform m_disparity_to_depth(false);

	cv::namedWindow("Filtered Depth Frame", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Color Frame", cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("Holes Fill", "Filtered Depth Frame", &holes_fill, 5, on_trackbar);
	cv::createTrackbar("Filter Magnitude", "Filtered Depth Frame", &filter_magnitude, 4, on_trackbar);
	cv::createTrackbar("Smooth Alpha", "Filtered Depth Frame", &smooth_alpha, 75, on_trackbar);
	cv::createTrackbar("Smooth Delta", "Filtered Depth Frame", &smooth_delta, 49, on_trackbar);
	cv::createTrackbar("Holes Fill Filter", "Filtered Depth Frame", &holes_fill_filter, 2, on_trackbar);

	while (true)
	{
		// 等待下一帧数据
		rs2::frameset frameset = pipe.wait_for_frames();
		rs2::align align2color(RS2_STREAM_COLOR);
		frameset = align2color.process(frameset);

		// 更新空间滤波器参数
		m_spatial_filter.set_option(RS2_OPTION_HOLES_FILL, holes_fill);
		m_spatial_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, filter_magnitude + 1);
		m_spatial_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.25 + smooth_alpha / 100.0f);
		m_spatial_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, smooth_delta + 1);
		m_hole_fill_filter.set_option(RS2_OPTION_HOLES_FILL, holes_fill_filter);

		rs2::frame depth_frame = frameset.get_depth_frame()
									 .apply_filter(m_depth_to_disparity)
									 .apply_filter(m_spatial_filter)
									 .apply_filter(m_hole_fill_filter)
									 .apply_filter(m_disparity_to_depth);
		rs2::frame color_frame = frameset.get_color_frame();

		// 将RealSense帧转换为OpenCV矩阵
		cv::Mat depth_image(cv::Size(640, 480), CV_16UC1, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);
		cv::Mat color_image(cv::Size(640, 480), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);

		// 将深度图像归一化以适合显示
		cv::Mat depth_image_normalized;
		cv::normalize(depth_image, depth_image_normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

		// 显示深度和颜色图像
		cv::imshow("Filtered Depth Frame", depth_image_normalized);
		cv::imshow("Color Frame", color_image);

		// 检查用户输入以退出循环
		if (cv::waitKey(1) == 27)
		{ // 27 是 ESC 键的 ASCII 码
			break;
		}
	}
}

void keyboardEventOccurred_test(const pcl::visualization::KeyboardEvent &event, void *viewer_void)
{
	pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);
	if (event.getKeySym() == "q" && event.keyDown())
	{
		viewer->close();
	}
}

void testIntrinsic(FetchInterface::Ptr fetcher, Intrinsic &rgb_intrinsic, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	ConfigParser &config = ConfigParser::Instance();
	cv::Mat depth_img, rgb_img;
	for (int i = 0; i < config.num_frames(); i++)
	{
		LOG(INFO) << "The " << i << "th Frame";
		fetcher->FetchDepthAndRGBImage(i, depth_img, rgb_img);
		float fx = rgb_intrinsic.focal_x;
		float fy = rgb_intrinsic.focal_y;
		float cx = rgb_intrinsic.principal_x;
		float cy = rgb_intrinsic.principal_y;
		LOG(INFO) << "fx: " << fx << " fy: " << fy << " cx: " << cx << " cy: " << cy;
		LOG(INFO) << "depth_img rows: " << depth_img.rows << " depth_img cols: " << depth_img.cols;
		float depth_scale = 0.001;
		for (int row = 0; row < depth_img.rows; row++)
		{
			for (int col = 0; col < depth_img.cols; col++)
			{
				float z = depth_img.at<ushort>(row, col) * depth_scale;
				if (z <= 0)
					continue;

				float x = (col - cx) * z / fx;
				float y = (row - cy) * z / fy;

				pcl::PointXYZRGB point;
				point.x = x;
				point.y = y;
				point.z = z;
				point.r = rgb_img.at<cv::Vec3b>(row, col)[2];
				point.g = rgb_img.at<cv::Vec3b>(row, col)[1];
				point.b = rgb_img.at<cv::Vec3b>(row, col)[0];

				cloud->points.push_back(point);
			}
		}
		LOG(INFO) << "size of cloud: " << cloud->size();

		pcl::visualization::PCLVisualizer viewer("viewer");
		viewer.addPointCloud<pcl::PointXYZRGB>(cloud);
		viewer.addCoordinateSystem(0.5);

		viewer.registerKeyboardCallback(keyboardEventOccurred_test, (void *)&viewer);
		while (!viewer.wasStopped())
		{
			viewer.spinOnce(100);
		}
		cloud->clear();
	}
}

//  测试获取图像，无限帧
void test_infinite(FetchInterface::Ptr fetcher)
{
	while (true)
	{
		cv::Mat depth_img, rgb_img;
		fetcher->FetchDepthAndRGBImage(0, depth_img, rgb_img);
		depth_img.convertTo(depth_img, CV_8UC1, 255.0 / 4000); // 假设深度范围是0-10000mm
		cv::Mat depth_colormap;
		cv::applyColorMap(depth_img, depth_colormap, cv::COLORMAP_JET);
		cv::namedWindow("depth", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("rgb", cv::WINDOW_AUTOSIZE);
		cv::imshow("depth", depth_colormap);
		cv::imshow("rgb", rgb_img);
		if (cv::waitKey(1) == 27)
		{
			break;
		}
	}
}

// 测试获取图像，有限帧
void test_finite(FetchInterface::Ptr fetcher, ConfigParser &config)
{
	for (int i = 0; i < config.num_frames(); i++)
	{
		LOG(INFO) << "The " << i << "th Frame";
		cv::Mat depth_img, rgb_img;
		fetcher->FetchDepthAndRGBImage(i, depth_img, rgb_img);
		depth_img.convertTo(depth_img, CV_8UC1, 255.0 / 4000); // 假设深度范围是0-10000mm
		cv::Mat depth_colormap;
		cv::applyColorMap(depth_img, depth_colormap, cv::COLORMAP_JET);
		cv::namedWindow("depth", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("rgb", cv::WINDOW_AUTOSIZE);
		cv::imshow("depth", depth_colormap);
		cv::imshow("rgb", rgb_img);
		if (cv::waitKey(1) == 27)
		{
			break;
		}
	}
}

void test_imgproc(std::string &config_path)
{
	// Parpare the test data

	auto &config = ConfigParser::Instance();
	config.ParseConfig(config_path);

	// First test fetching
	FetchInterface::Ptr fetcher;
	if (config.getIOMode() == "GenericFileFetch")
	{
		fetcher = std::make_shared<GenericFileFetch>(config.data_path());
	}
	else if (config.getIOMode() == "VolumeDeformFileFetch")
	{
		fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());
	}
	else if (config.getIOMode() == "realsense")
	{
		fetcher = std::make_shared<RealsenseFetch>(config.data_path(), config.isSaveOnlineFrame());
	}
	else
	{
		throw(std::runtime_error(config.getIOMode() + " io_mode not supported"));
	}

	ImageProcessor::Ptr processor = std::make_shared<ImageProcessor>(fetcher);
	Intrinsic rgb_intrinsic = config.rgb_intrinsic_raw();
	LOG(INFO) << "image rows: " << config.raw_image_rows() << " image cols: " << config.raw_image_cols() << " fy: " << rgb_intrinsic.focal_y << " fx: " << rgb_intrinsic.focal_x << " cx: " << rgb_intrinsic.principal_x << " cy: " << rgb_intrinsic.principal_y;

	// 测试点云生成
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	cloud->width = config.raw_image_cols() * config.raw_image_rows();
	cloud->height = 1;

	// test_infinite(fetcher);
	// test_finite(fetcher, config);
	// testIntrinsic(fetcher, rgb_intrinsic, cloud);

	// 图像处理结果显示， 可视化点云与分割结果
	CameraObservation observation;
	for (int i = 0; i < config.num_frames(); i++)
	{
		LOG(INFO) << "The " << i << "th Frame";
		processor->ProcessFrameStreamed(observation, i);

		// 	/* auto draw_func = [&]() {
		// 	Visualizer::DrawColoredPointCloud(observation.vertex_confid_map, observation.normalized_rgba_map);
		// 	// Visualizer::DrawNormalizeRGBImage(observation.normalized_rgba_map);
		// 	// Visualizer::DrawSegmentMask(observation.foreground_mask, observation.normalized_rgba_map, 1);
		// 	//Visualizer::DrawGrayScaleImage(observation.filter_foreground_mask);
		// 	};

		// 	std::thread draw_thread(draw_func);
		// 	draw_thread.join(); */
		Visualizer::DrawColoredPointCloud(observation.vertex_confid_map, observation.normalized_rgba_map);
		Visualizer::DrawSegmentMask(observation.foreground_mask, observation.normalized_rgba_map, 1);
	}
}

int main(int argc, char **argv)
{
	std::string config_path;
	if (argc == 2)
	{
		config_path = argv[1];
	}
	else
	{
		config_path = "/home/roboticlab/Desktop/Maihn/surfelwarp/test_data/realsense_config.json";
	}
	test_imgproc(config_path);
	// testFilter();
}