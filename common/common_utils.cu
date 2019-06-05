#include "common/common_utils.h"
#include "common/safe_call_utils.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <driver_types.h>


CUcontext surfelwarp::initCudaContext(int selected_device) {
    //Initialize the cuda driver api
    cuSafeCall(cuInit(0));

    //Query the device
    int device_count = 0;
    cuSafeCall(cuDeviceGetCount(&device_count));
    for(auto dev_idx = 0; dev_idx < device_count; dev_idx++) {
        char dev_name[256] = { 0 };
        cuSafeCall(cuDeviceGetName(dev_name, 256, dev_idx));
        printf("device %d: %s\n", dev_idx, dev_name);
    }

    //Select the device
    printf("device %d is used as parallel processor.\n", selected_device);
    CUdevice cuda_device;
    cuSafeCall(cuDeviceGet(&cuda_device, selected_device));

    //Create cuda context
    CUcontext cuda_context;
    cuSafeCall(cuCtxCreate(&cuda_context, CU_CTX_SCHED_AUTO, cuda_device));
    return cuda_context;
}

void surfelwarp::destroyCudaContext(CUcontext context) {
    cudaDeviceSynchronize();
    cuSafeCall(cuCtxDestroy(context));
}






