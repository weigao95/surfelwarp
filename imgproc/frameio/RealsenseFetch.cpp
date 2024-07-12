#include "RealsenseFetch.h"

surfelwarp::RealsenseFetch::RealsenseFetch(
    const path& data_path, bool save_online_frame)
: m_save_online_frame(save_online_frame)
, save_folder(data_path)
// , m_depth_to_disparity(true)
// , m_disparity_to_depth(false)
{
    m_configuration.disable_all_streams();
    m_configuration.enable_stream(
        RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 60);
    m_configuration.enable_stream(
        RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    auto devices = m_ctx.query_devices();
    if (devices.size() == 0)
        throw std::runtime_error {"No realsense device detected!"};

    {
        auto device = devices[0]; // The device handle defined here is invalid
                                  // after pipeline.start()

        // Print info about the device
        std::cout << "=================Realsense Info=================="
                  << std::endl;
        std::cout << "Using this RealSense device:" << std::endl;
        for (int info_idx = 0;
             info_idx < static_cast<int>(RS2_CAMERA_INFO_COUNT); ++info_idx) {
            auto info_type = static_cast<rs2_camera_info>(info_idx);
            std::cout << "  " << std::left << std::setw(20) << info_type
                      << " : ";
            if (device.supports(info_type))
                std::cout << device.get_info(info_type) << std::endl;
            else
                std::cout << "Not supported" << std::endl;
        }
        std::cout << "=================Realsense Info=================="
                  << std::endl;
    }

    m_profile = m_pipe.start(m_configuration);
    // 获取设备并设置高精度模式
    rs2::device dev = m_profile.get_device();
    rs2::depth_sensor depth_sensor = dev.first<rs2::depth_sensor>();

    if (depth_sensor.supports(RS2_OPTION_VISUAL_PRESET)) {
        depth_sensor.set_option(
            RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY);
    } else {
        std::cerr << "This device does not support visual presets."
                  << std::endl;
    }

    m_spatial_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);
    m_spatial_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.5);
    m_spatial_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20);
    m_temporal_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 1);
    m_temporal_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 40);
    m_dec_filter.set_option(
        RS2_OPTION_FILTER_MAGNITUDE, config.downsample_scale());

#if defined(DEBUG)
    auto streams = m_pipe.get_active_profile().get_streams();
    for (const auto& stream : streams) {
        if (stream.stream_type() == RS2_STREAM_COLOR) {
            if (config.use_downsample()) {
                m_rgb_intrinsics
                    = stream.as<rs2::video_stream_profile>().get_intrinsics();
                std::cout << "rgb_intrinsics.focal_x: "
                          << m_rgb_intrinsics.fx / config.downsample_scale()
                          << std::endl;
                std::cout << "rgb_intrinsics.focal_y: "
                          << m_rgb_intrinsics.fy / config.downsample_scale()
                          << std::endl;
                std::cout << "rgb_intrinsics.image_height: "
                          << m_rgb_intrinsics.height / config.downsample_scale()
                          << std::endl;
                std::cout << "rgb_intrinsics.image_width: "
                          << m_rgb_intrinsics.width / config.downsample_scale()
                          << std::endl;
                std::cout << "rgb_intrinsics.principal_x: "
                          << m_rgb_intrinsics.ppx / config.downsample_scale()
                          << std::endl;
                std::cout << "rgb_intrinsics.principal_y: "
                          << m_rgb_intrinsics.ppy / config.downsample_scale()
                          << std::endl;
            } else {
                m_rgb_intrinsics
                    = stream.as<rs2::video_stream_profile>().get_intrinsics();
                std::cout << "rgb_intrinsics.focal_x: " << m_rgb_intrinsics.fx
                          << std::endl;
                std::cout << "rgb_intrinsics.focal_y: " << m_rgb_intrinsics.fy
                          << std::endl;
                std::cout << "rgb_intrinsics.image_height: "
                          << m_rgb_intrinsics.height << std::endl;
                std::cout << "rgb_intrinsics.image_width: "
                          << m_rgb_intrinsics.width << std::endl;
                std::cout << "rgb_intrinsics.principal_x: "
                          << m_rgb_intrinsics.ppx << std::endl;
                std::cout << "rgb_intrinsics.principal_y: "
                          << m_rgb_intrinsics.ppy << std::endl;
            }
        }
    }
#endif

    if (m_save_online_frame) {
        if (!boost::filesystem::exists(save_folder)) {
            boost::filesystem::create_directories(save_folder);
        }
    }
}

surfelwarp::RealsenseFetch::~RealsenseFetch()
{
    m_pipe.stop();
}

void surfelwarp::RealsenseFetch::FetchDepthImage(
    size_t frame_idx, cv::Mat& depth_img)
{
    grab_frame();
    depth_img = m_depth_image;
}

void surfelwarp::RealsenseFetch::FetchDepthImage(
    size_t frame_idx, void* depth_img)
{
}

void surfelwarp::RealsenseFetch::FetchRGBImage(
    size_t frame_idx, cv::Mat& rgb_img)
{
    grab_frame();
    rgb_img = m_depth_image;
}

void surfelwarp::RealsenseFetch::FetchRGBImage(size_t frame_idx, void* rgb_img)
{
}

void surfelwarp::RealsenseFetch::FetchDepthAndRGBImage(
    size_t frame_idx, cv::Mat& depth_img, cv::Mat& rgb_img)
{
    grab_frame();
    rgb_img = m_color_image;
    depth_img = m_depth_image;
}

void surfelwarp::RealsenseFetch::grab_frame()
{
    try {
        rs2::disparity_transform depth_to_disparity = true;
        rs2::disparity_transform disparity_to_depth = false;
        // 等待并获取新的帧数据
        rs2::frameset frameset = m_pipe.wait_for_frames();

        // 对齐深度帧到彩色帧
        rs2::align align2color(RS2_STREAM_COLOR);
        frameset = align2color.process(frameset);

        // 获取对齐后的深度帧和彩色帧
        m_depth_frame = frameset.get_depth_frame()
                            .apply_filter(m_dec_filter)
                            .apply_filter(depth_to_disparity)
                            .apply_filter(m_spatial_filter)
                            .apply_filter(m_temporal_filter)
                            .apply_filter(disparity_to_depth);

        m_color_frame = frameset.get_color_frame();

        // 确保深度帧和彩色帧不为空
        if (!m_depth_frame || !m_color_frame) {
            throw std::runtime_error(
                "Failed to get frames from RealSense camera.");
        }

        // 将深度帧和彩色帧转换为 OpenCV 矩阵
        m_color_image = cv::Mat(cv::Size(640, 480), CV_8UC3,
            (void*)m_color_frame.get_data(), cv::Mat::AUTO_STEP);
        if (config.use_downsample()) {
            m_depth_image = cv::Mat(cv::Size(320, 240), CV_16UC1,
                (void*)m_depth_frame.get_data(), cv::Mat::AUTO_STEP);
            DownSampleImage();
        } else {
            m_depth_image = cv::Mat(cv::Size(640, 480), CV_16UC1,
                (void*)m_depth_frame.get_data(), cv::Mat::AUTO_STEP);
        }

        // 打印矩阵信息用于调试
        // #if defined(DEBUG)
        //     LOG(INFO)<<"Depth image: " << m_depth_image.size() << " " <<
        //     m_depth_image.type(); LOG(INFO)<<"Color image: " <<
        //     m_color_image_downsampled.size() << " " << m_color_image.type();
        // #endif

        // 克隆矩阵并添加到相应的向量中，确保每次添加的是独立的副本
        m_depth_image_vec.push_back(m_depth_image.clone());
        m_color_image_vec.push_back(m_color_image.clone());

        // 如果需要保存图像，取消注释以下代码并添加保存逻辑
        if (m_save_online_frame) {
            std::string file_name = save_folder.string() + "/frame-";
            char frame_idx_str[20];
            sprintf(frame_idx_str, "%06d", static_cast<int>(m_cur_frame_num));
            file_name += frame_idx_str;
            cv::imwrite(file_name + ".depth.png", m_depth_image.clone());
            cv::imwrite(file_name + ".color.png", m_color_image.clone());
        }

        // 增加当前帧编号
        m_cur_frame_num++;
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cerr << "Unknown exception occurred." << std::endl;
        throw;
    }
}

// 彩色图像降采样
void surfelwarp::RealsenseFetch::DownSampleImage()
{
    d_color_src = cv::cuda::createContinuous(
        config.raw_image_rows(), config.raw_image_cols(), CV_8UC3);
    d_color_src.upload(m_color_image);
    m_color_image.release();
    cv::cuda::pyrDown(d_color_src, d_color_downsampled);
    d_color_downsampled.download(m_color_image);
}