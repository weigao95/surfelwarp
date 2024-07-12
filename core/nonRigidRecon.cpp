#include "nonRigidRecon.h"

surfelwarp::nonRigidRecon::nonRigidRecon()
: m_config_path(
      "/home/roboticlab/Desktop/Maihn/teleop_virtual_fixture/thirdparty/"
      "surfelwarp/test_data/realsense_config.json")
, config(ConfigParser::Instance())
, current_frame_idx(1)
{
    config.ParseConfig(m_config_path);
    surfelWarp = std::make_shared<SurfelWarpSerial>();
}

surfelwarp::nonRigidRecon::nonRigidRecon(std::string config_path)
: m_config_path(config_path)
, config(ConfigParser::Instance())
{
    config.ParseConfig(m_config_path);
    surfelWarp = std::make_shared<SurfelWarpSerial>();
}

surfelwarp::nonRigidRecon::~nonRigidRecon() { }

void surfelwarp::nonRigidRecon::launch()
{
    if (surfelWarp) {
        surfelWarp->ProcessFirstFrame();
        while (!stop) {
            {
                std::lock_guard<std::mutex> lock(surfel_geometry_mutex);
                surfelWarp->ProcessNextFrameWithReinit(config);
                new_pcd_ready = true;
                current_frame_idx++;
                new_pcd_cv.notify_one();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } else {
        LOG(ERROR) << "surfelWarp is not initialized!";
    }
}

std::vector<float4> surfelwarp::nonRigidRecon::GetPointCloud()
{
    std::unique_lock<std::mutex> lock(surfel_geometry_mutex);
    new_pcd_cv.wait(lock, [this] { return new_pcd_ready.load(); });
    std::vector<float4> pcd;
    auto surfel_geometry = surfelWarp->GetSurfelGeometry();
    surfel_geometry->GetLiveVertexConfidence().Download(pcd);
    new_pcd_ready = false;
    return pcd;
}
