#include "common/encode_utils.h"
#include "common/Constants.h"
#include "math/vector_ops.hpp"
#include "core/geometry/VoxelSubsamplerSorting.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__global__ void createVoxelKeyKernel(
		DeviceArrayView<float4> points,
		int* encoded_voxel_key,
		const float voxel_size
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= points.Size()) return;
		const int voxel_x = __float2int_rd(points[idx].x / voxel_size);
		const int voxel_y = __float2int_rd(points[idx].y / voxel_size);
		const int voxel_z = __float2int_rd(points[idx].z / voxel_size);
		const int encoded = encodeVoxel(voxel_x, voxel_y, voxel_z);
		encoded_voxel_key[idx] = encoded;
	}

	__global__ void labelSortedVoxelKeyKernel(
		const PtrSz<const int> sorted_voxel_key,
		unsigned* key_label
	) {
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx == 0) key_label[0] = 1;
		else {
			if (sorted_voxel_key[idx] != sorted_voxel_key[idx - 1])
				key_label[idx] = 1;
			else
				key_label[idx] = 0;
		}
	}


	__global__ void compactedVoxelKeyKernel(
		const PtrSz<const int> sorted_voxel_key,
		const unsigned* voxel_key_label,
		const unsigned* prefixsumed_label,
		int* compacted_key,
		DeviceArraySlice<int> compacted_offset
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= sorted_voxel_key.size) return;
		if (voxel_key_label[idx] == 1) {
			compacted_key[prefixsumed_label[idx] - 1] = sorted_voxel_key[idx];
			compacted_offset[prefixsumed_label[idx] - 1] = idx;
		}
		if (idx == 0) {
			compacted_offset[compacted_offset.Size() - 1] = sorted_voxel_key.size;
		}
	}

	__global__ void samplingPointsKernel(
		const DeviceArrayView<int> compacted_key,
		const int* compacted_offset,
		const float4* sorted_points,
		const float voxel_size,
		float4* sampled_points
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= compacted_key.Size()) return;
		// The voxel position
		const int encoded = compacted_key[idx];
		int x, y, z;
		decodeVoxel(encoded, x, y, z);
		const float3 voxel_center = make_float3(
			float(x + 0.5) * voxel_size,
			float(y + 0.5) * voxel_size,
			float(z + 0.5) * voxel_size
		);

		// Find the one closed to the center
		float min_dist_square = 1e5;
		int min_dist_idx = compacted_offset[idx];
		for (int i = compacted_offset[idx]; i < compacted_offset[idx + 1]; i++) {
			const float4 point4 = sorted_points[i];
			const float3 point = make_float3(point4.x, point4.y, point4.z);
			const float new_dist = squared_norm(point - voxel_center);
			if (new_dist < min_dist_square) {
				min_dist_square = new_dist;
				min_dist_idx = i;
			}
		}

		// Store the result to global memory
		sampled_points[idx] = sorted_points[min_dist_idx];
	}

}; // namespace device
}; // namespace surfelwarp


void surfelwarp::VoxelSubsamplerSorting::AllocateBuffer(unsigned max_input_points) {
	m_point_key.AllocateBuffer(max_input_points);
	m_point_key_sort.AllocateBuffer(max_input_points);
	m_voxel_label.AllocateBuffer(max_input_points);
	m_voxel_label_prefixsum.AllocateBuffer(max_input_points);
	
	const auto compacted_max_size = max_input_points / 5;
	m_compacted_voxel_key.AllocateBuffer(compacted_max_size);
	m_compacted_voxel_offset.AllocateBuffer(compacted_max_size);
	m_subsampled_point.AllocateBuffer(compacted_max_size);
}

void surfelwarp::VoxelSubsamplerSorting::ReleaseBuffer() {
	//Constants::kMaxNumSurfels
	m_point_key.ReleaseBuffer();
	m_voxel_label.ReleaseBuffer();
	
	//smaller buffer
	m_compacted_voxel_key.ReleaseBuffer();
	m_compacted_voxel_offset.ReleaseBuffer();
	m_subsampled_point.ReleaseBuffer();
}

surfelwarp::DeviceArrayView<float4> surfelwarp::VoxelSubsamplerSorting::PerformSubsample(
	const surfelwarp::DeviceArrayView<float4> &points,
	const float voxel_size,
	cudaStream_t stream
) {
	buildVoxelKeyForPoints(points, voxel_size, stream);
	sortCompactVoxelKeys(points, stream);
	collectSubsampledPoint(m_subsampled_point, voxel_size, stream);
	return m_subsampled_point.ArrayView();
}

void surfelwarp::VoxelSubsamplerSorting::PerformSubsample(
	const surfelwarp::DeviceArrayView<float4> &points,
	surfelwarp::SynchronizeArray<float4> &subsampled_points,
	const float voxel_size,
	cudaStream_t stream
) {
	buildVoxelKeyForPoints(points, voxel_size, stream);
	sortCompactVoxelKeys(points, stream);
	collectSynchronizeSubsampledPoint(subsampled_points, voxel_size, stream);
}

void surfelwarp::VoxelSubsamplerSorting::buildVoxelKeyForPoints(
	const surfelwarp::DeviceArrayView<float4> &points,
	const float voxel_size,
	cudaStream_t stream
) {
	//Correct the size of arrays
	m_point_key.ResizeArrayOrException(points.Size());
	
	//Call the method
	dim3 blk(256);
	dim3 grid(divUp(points.Size(), blk.x));
	device::createVoxelKeyKernel<<<grid, blk, 0, stream>>>(
		points,
		m_point_key,
		voxel_size
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::VoxelSubsamplerSorting::sortCompactVoxelKeys(
	const DeviceArrayView<float4>& points,
	cudaStream_t stream
) {
	//Perform sorting
	m_point_key_sort.Sort(m_point_key.ArrayReadOnly(), points, stream);
	
	//Label the sorted keys
	m_voxel_label.ResizeArrayOrException(points.Size());
	dim3 blk(128);
	dim3 grid(divUp(points.Size(), blk.x));
	device::labelSortedVoxelKeyKernel<<<grid, blk, 0, stream>>>(
		m_point_key_sort.valid_sorted_key,
		m_voxel_label.ArraySlice()
	);
	
	//Prefix sum
	m_voxel_label_prefixsum.InclusiveSum(m_voxel_label.ArrayView(), stream);
	//cudaSafeCall(cudaStreamSynchronize(stream));
	
	//Query the number of voxels
	unsigned num_voxels;
	const auto& prefixsum_label = m_voxel_label_prefixsum.valid_prefixsum_array;
	cudaSafeCall(cudaMemcpyAsync(
		&num_voxels,
		prefixsum_label.ptr() + prefixsum_label.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream
	));
	cudaSafeCall(cudaStreamSynchronize(stream));
	
	//Construct the compacted array
	m_compacted_voxel_key.ResizeArrayOrException(num_voxels);
	m_compacted_voxel_offset.ResizeArrayOrException(num_voxels + 1);
	device::compactedVoxelKeyKernel<<<grid, blk, 0, stream>>>(
		m_point_key_sort.valid_sorted_key,
		m_voxel_label,
		prefixsum_label,
		m_compacted_voxel_key,
		m_compacted_voxel_offset.ArraySlice()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::VoxelSubsamplerSorting::collectSubsampledPoint(
	surfelwarp::DeviceBufferArray<float4> &subsampled_points,
	const float voxel_size,
	cudaStream_t stream
) {
	//Correct the size
	const auto num_voxels = m_compacted_voxel_key.ArraySize();
	subsampled_points.ResizeArrayOrException(num_voxels);
	
	//Everything is ok
	dim3 sample_blk(128);
	dim3 sample_grid(divUp(num_voxels, sample_blk.x));
	device::samplingPointsKernel<<<sample_grid, sample_blk, 0, stream>>>(
		m_compacted_voxel_key.ArrayView(),
		m_compacted_voxel_offset,
		m_point_key_sort.valid_sorted_value,
		voxel_size,
		subsampled_points
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::VoxelSubsamplerSorting::collectSynchronizeSubsampledPoint(
	surfelwarp::SynchronizeArray<float4> &subsampled_points,
	const float voxel_size,
	cudaStream_t stream
) {
	//Correct the size
	const auto num_voxels = m_compacted_voxel_key.ArraySize();
	subsampled_points.ResizeArrayOrException(num_voxels);
	
	//Hand on it to device
	auto subsampled_points_slice = subsampled_points.DeviceArrayReadWrite();
	dim3 sample_blk(128);
	dim3 sample_grid(divUp(num_voxels, sample_blk.x));
	device::samplingPointsKernel<<<sample_grid, sample_blk, 0, stream>>>(
		m_compacted_voxel_key.ArrayView(),
		m_compacted_voxel_offset,
		m_point_key_sort.valid_sorted_value,
		voxel_size,
		subsampled_points_slice
	);
	
	//Sync it to host
	subsampled_points.SynchronizeToHost(stream);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

