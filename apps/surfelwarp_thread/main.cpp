#include "core/nonRigidRecon.h"
#include <future>
#include <chrono>

using namespace surfelwarp;

void nonRigidReconTask(
    std::promise<nonRigidRecon::Ptr>& recon_promise, std::mutex& output_mutex)
{
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << "================================================="
                  << std::endl;
        std::cout << "           Non-rigid Recon Thread Start" << std::endl;
        std::cout << "================================================="
                  << std::endl;
    }

    auto recon = std::make_shared<nonRigidRecon>();
    recon_promise.set_value(recon);
    recon->launch();
}

void getPointCloudTask(
    std::future<nonRigidRecon::Ptr>& recon_future, std::mutex& output_mutex)
{
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << "================================================="
                  << std::endl;
        std::cout << "              Get PCD Thread Start" << std::endl;
        std::cout << "================================================="
                  << std::endl;
    }

    auto recon = recon_future.get();
    while (!recon->IsStopped()) {
        auto pcd = recon->GetPointCloud();
        if (!pcd.empty()) {
            std::cout << "================Got Point Ploud=================="
                      << std::endl;
        } else {
            std::cout << "===========Couldn't Get Point Cloud=============="
                      << std::endl;
        }
    }
}

int main()
{
    std::mutex output_mutex;
    std::promise<nonRigidRecon::Ptr> recon_promise;
    std::future<nonRigidRecon::Ptr> recon_future = recon_promise.get_future();

    std::thread nonrigid_recon_thread(
        nonRigidReconTask, std::ref(recon_promise), std::ref(output_mutex));
    std::thread get_pcd_thread(
        getPointCloudTask, std::ref(recon_future), std::ref(output_mutex));

    nonrigid_recon_thread.join();
    get_pcd_thread.join();

    return 0;
}