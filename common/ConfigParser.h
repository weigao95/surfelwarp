#pragma once
#include "common/common_types.h"
#include "math/device_mat.h"
#include <Eigen/Eigen>
#include <string>
#include <boost/filesystem.hpp>

namespace surfelwarp {

class ConfigParser
{
private:
    // Do not allow user-contruct
    explicit ConfigParser();
    void setDefaultParameters();

public:
    static ConfigParser& Instance();
    void ParseConfig(const std::string& config_path);
    void SaveConfig(const std::string& config_path) const;

    /* The path information
     */
private:
    std::string m_data_prefix;
    std::string m_gpc_model_path;
    std::string m_save_path;
    void setDefaultPathConfig();
    void savePathConfigToJson(void* json_ptr) const;
    void loadPathConfigFromJson(const void* json_ptr);

public:
    const boost::filesystem::path data_path() const;
    const boost::filesystem::path gpc_model_path() const;
    const boost::filesystem::path save_path() const;

    /* The frame index
     */
private:
    int m_start_frame_idx;
    int m_num_frames;
    void setDefaultFrameIndex();
    void saveFrameIndexToJson(void* json_ptr) const;
    void loadFrameIndexFromJson(const void* json_ptr);

public:
    int start_frame_idx() const;
    int num_frames() const;

    // The frame peroids for reinit and recent
private:
    bool m_use_periodic_reinit;
    int m_reinit_period;
    void setDefaultPeroidsValue();
    void savePeroidsValueToJson(void* json_ptr) const;
    void loadPeroidsValueFromJson(const void* json_ptr);

public:
    bool use_periodic_reinit() const { return m_use_periodic_reinit; }
    int reinit_period() const { return m_reinit_period; }

    /* The method and member about the size of image
     */
private:
    unsigned m_raw_image_rows;
    unsigned m_raw_image_cols;
    unsigned m_clip_image_rows;
    unsigned m_clip_image_cols;
    void setDefaultImageSize();
    void saveImageSizeToJson(void* json_ptr) const;
    void loadImageSizeFromJson(const void* json_ptr);

public:
    unsigned raw_image_rows() const { return m_raw_image_rows; }
    unsigned raw_image_cols() const { return m_raw_image_cols; }
    unsigned clip_image_rows() const { return m_clip_image_rows; }
    unsigned clip_image_cols() const { return m_clip_image_cols; }

    /* The method and member about cliping
     */
private:
    unsigned m_clip_near;
    unsigned m_clip_far;
    void setDefaultClipValue();
    void saveClipValueToJson(void* json_ptr) const;
    void loadClipValueFromJson(const void* json_ptr);

public:
    unsigned clip_near_mm() const { return m_clip_near; }
    unsigned clip_far_mm() const { return m_clip_far; }
    float max_rendering_depth() const { return clip_far_meter(); }
    float clip_far_meter() const
    {
        float clip_far_meters = float(m_clip_far) / 1000.0f;
        return clip_far_meters;
    }

    /* The method for intrinsic querying
     */
private:
    Intrinsic raw_depth_intrinsic;
    Intrinsic raw_rgb_intrinsic;
    Intrinsic clip_rgb_intrinsic;
    const int m_downsample_scale = 2;
    bool m_use_downsample;
    void setDefaultCameraIntrinsic();
    void saveCameraIntrinsicToJson(void* json_ptr) const;
    void loadCameraIntrinsicFromJson(const void* json_ptr);
    void loadDownSample(const void* json_ptr);

public:
    Intrinsic depth_intrinsic_raw() const;
    Intrinsic rgb_intrinsic_raw() const;
    Intrinsic rgb_intrinsic_clip() const;
    mat34 depth2rgb_dev() const;
    bool use_downsample() const { return m_use_downsample; }
    int downsample_scale() const { return m_downsample_scale; }

    // A various of configs for penalty constants
    // m_use_segmentation: whether to use segmentation
private:
    bool m_use_density_term;
    bool m_use_foreground_term;
    bool m_use_offline_foreground;
    bool m_use_segmentation;
    void setDefaultPenaltyConfigs();
    void savePenaltyConfigToJson(void* json_ptr) const;
    void loadPenaltyConfigFromJson(const void* json_ptr);

public:
    bool use_foreground_term() const;
    bool use_offline_foreground_segmneter() const;
    bool use_density_term() const;
    bool use_segmentation() const;

    // add `io_mode` for kinectdk, and realsense
    // default: io_mode = "local_file", surfelwarp_app will run on this io_mode,
    //          where imiages are read from local file
    // for kinectdk: io_mode = "kinect_dk", surfelwarp_kinectdk will run on this
    // io_mode,
    //               where images are captured from kinect dk at runing-time;
private:
    std::string m_io_mode;
    bool m_save_online_frame;
    void setDefaultIOMode();
    void loadIOModeFromJson(const void* json_ptr);
    void loadSaveOnlineFrame(const void* json_ptr);
    void saveIOModeToJson(void* json_ptr) const;

public:
    std::string getIOMode() const;
    bool isSaveOnlineFrame() const;

private:
    bool m_offline_rendering;
    bool m_show_online;
    void setDefaultOfflineRendering();
    void loadOfflineRendering(const void* json_ptr);
    void saveOfflineRendering(void* json_ptr) const;

public:
    bool isShowOnline() const;
    bool isOfflineRendering() const;
    void setOfflineRendering(bool offline_rendering);
    // set public data member for convenience
    // save camera observation
    bool save_segment_mask;
    bool save_filter_depth_image;
    bool save_raw_depth_image;
    // save solver maps
    bool save_rendered_albedo_map;
    bool save_validity_index_map;
    bool save_alignment_error_map;
    // save visualization maps
    bool save_live_normal_map;
    bool save_live_albedo_map;
    bool save_live_phong_map;
    bool save_reference_normal_map;
    bool save_reference_albedo_map;
    bool save_reference_phong_map;
    // save se3
    bool save_se3;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}
