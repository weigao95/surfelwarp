//
// Created by wei on 3/23/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/SynchronizeArray.h"
#include <memory>

namespace surfelwarp {
	
	class VoxelSubsampler {
	public:
		using Ptr = std::shared_ptr<VoxelSubsampler>;
		VoxelSubsampler() = default;
		virtual ~VoxelSubsampler() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(VoxelSubsampler);
		
		//Again, explicit malloc
		virtual void AllocateBuffer(unsigned max_input_points) = 0;
		virtual void ReleaseBuffer() = 0;
		
		//The interface functions
		virtual DeviceArrayView<float4> PerformSubsample(
			const DeviceArrayView<float4>& points,
			const float voxel_size,
			cudaStream_t stream = 0
		) = 0;
		
		virtual void PerformSubsample(
			const DeviceArrayView<float4>& points,
			SynchronizeArray<float4>& subsampled_points,
			const float voxel_size,
			cudaStream_t stream = 0
		) = 0;
	};
	
}