#include "common/common_utils.h"
#include "common/ConfigParser.h"
#include "core/SurfelWarpSerial.h"
#include <boost/filesystem.hpp>
#include <condition_variable>
#include <atomic>

#ifndef NONRIGIDRECON_H_
#define NONRIGIDRECON_H_

namespace surfelwarp {

class nonRigidRecon
{
public:
    using Ptr = std::shared_ptr<nonRigidRecon>;
    nonRigidRecon();
    nonRigidRecon(std::string config_path);
    SURFELWARP_NO_COPY_ASSIGN_MOVE(nonRigidRecon);
    ~nonRigidRecon();

    void launch();
    std::vector<float4> GetPointCloud();
    bool IsStopped() const { return stop; }

private:
    // TODO: provide camera pose by flexiv robots
    SurfelWarpSerial::Ptr surfelWarp;
    ConfigParser& config;
    std::string m_config_path;
    bool stop = false;
    size_t current_frame_idx;

    // thread
    std::mutex surfel_geometry_mutex;
    std::condition_variable new_pcd_cv;
    std::atomic<bool> new_pcd_ready {false};
};

}

#endif /* NONRIGIDRECON_H_*/
