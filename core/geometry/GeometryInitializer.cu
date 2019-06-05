//
// Created by wei on 3/19/18.
//

#include "common/Stream.h"
#include "common/Serializer.h"
#include "common/BinaryFileStream.h"
#include "core/geometry/GeometryInitializer.h"
#include "core/geometry/VoxelSubsamplerSorting.h"
#include "core/geometry/KNNSearch.h"
#include "core/geometry/WarpFieldUpdater.h"
#include "common/Constants.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__global__ void initializerCollectDepthSurfelKernel(
		DeviceArrayView<DepthSurfel> surfel_array,
		float4* reference_vertex_confid,
		float4* reference_normal_radius,
		float4* live_vertex_confid,
		float4* live_normal_radius,
		float4* color_time
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < surfel_array.Size()) {
			const DepthSurfel& surfel = surfel_array[idx];
			reference_vertex_confid[idx] = live_vertex_confid[idx] = surfel.vertex_confid;
			reference_normal_radius[idx] = live_normal_radius[idx] = surfel.normal_radius;
			color_time[idx] = surfel.color_time;
		}
	}

}; // namespace device	
}; // namespace surfelwarp


void surfelwarp::GeometryInitializer::AllocateBuffer() {
	//These functionality has been moved to WarpFieldInitializer
	//m_vertex_subsampler.reset(new VoxelSubsamplerSorting());
	//m_vertex_subsampler->AllocateBuffer(Constants::kMaxNumSurfels);
	//m_node_candidates.AllocateBuffer(Constants::kMaxSubsampleFrom * Constants::kMaxNumNodes);
}

void surfelwarp::GeometryInitializer::ReleaseBuffer() {
}

void surfelwarp::GeometryInitializer::InitFromObservationSerial(
	surfelwarp::SurfelGeometry &geometry,
	const surfelwarp::DeviceArrayView<surfelwarp::DepthSurfel> &surfel_array,
	cudaStream_t stream
) {
	geometry.ResizeValidSurfelArrays(surfel_array.Size());
	
	//Init the geometry
	const auto geometry_attributes = geometry.Geometry();
	initSurfelGeometry(geometry_attributes, surfel_array, stream);
}


void surfelwarp::GeometryInitializer::initSurfelGeometry(
	GeometryAttributes geometry,
	const DeviceArrayView<DepthSurfel>& surfel_array,
	cudaStream_t stream
) {
	//Invoke the kernel
	dim3 blk(256);
	dim3 grid(divUp(surfel_array.Size(), blk.x));
	device::initializerCollectDepthSurfelKernel<<<grid, blk, 0, stream>>>(
		surfel_array, 
		geometry.reference_vertex_confid,
		geometry.reference_normal_radius, 
		geometry.live_vertex_confid, 
		geometry.live_normal_radius, 
		geometry.color_time
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}