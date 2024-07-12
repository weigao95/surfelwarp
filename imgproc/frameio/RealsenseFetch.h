#pragma once
#include "FetchInterface.h"
#include <string>
#include <boost/filesystem.hpp>
#include <librealsense2/rs.hpp>
#include "common/ConfigParser.h"
#include "common/logging.h"

namespace surfelwarp {
/**
 * \brief Utility for fetching depth & RGB frames from realsense camera
 */
class RealsenseFetch : public FetchInterface
{
public:
    using Ptr = std::shared_ptr<RealsenseFetch>;
    using path = boost::filesystem::path;

    // todo: load device config from file
    explicit RealsenseFetch(const path& data_path, bool save_online_frame);

    ~RealsenseFetch();
    void FetchDepthImage(size_t frame_idx, cv::Mat& depth_img) override;
    void FetchDepthImage(size_t frame_idx, void* depth_img) override;
    void FetchRGBImage(size_t frame_idx, cv::Mat& rgb_img) override;
    void FetchRGBImage(size_t frame_idx, void* rgb_img) override;
    void FetchDepthAndRGBImage(
        size_t frame_idx, cv::Mat& depth_img, cv::Mat& rgb_img) override;
    void DownSampleImage() override;

    void grab_frame();

private:
    rs2::config m_configuration;
    rs2::pipeline m_pipe;
    rs2::context m_ctx;
    rs2::spatial_filter m_spatial_filter;
    rs2::temporal_filter m_temporal_filter;
    rs2::hdr_merge m_hdr_merge_filter;
    rs2::decimation_filter m_dec_filter;
    // rs2::disparity_transform m_depth_to_disparity;
    // rs2::disparity_transform m_disparity_to_depth;
    rs2::pipeline_profile m_profile;
    rs2::frame m_depth_frame;
    rs2::frame m_color_frame;
    rs2_intrinsics m_rgb_intrinsics;

    // images
    cv::Mat m_depth_image;
    cv::Mat m_color_image;
    cv::cuda::GpuMat d_color_src;
    cv::cuda::GpuMat d_color_downsampled;

    std::vector<cv::Mat> m_depth_image_vec;
    std::vector<cv::Mat> m_color_image_vec;

    // frame property
    cv::Size m_frame_size;
    size_t m_cur_frame_num;

    // save the frame
    bool m_save_online_frame;
    path save_folder;

    // scale factor
    ConfigParser& config = ConfigParser::Instance();
};
}