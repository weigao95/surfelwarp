#include "math/device_mat.h"
#include "core/geometry/DoubleBufferCompactor.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	struct LiveSurfelKNNCompactionOutput {
		float4* live_vertex_confid;
		float4* live_normal_radius;
		float4* color_time;
		ushort4* surfel_knn;
		float4* surfel_knnweight;
	};
	

	__global__ void compactRemainingAndAppendedSurfelKNNKernel(
		const AppendedObservationSurfelKNN appended_observation,
		const RemainingLiveSurfel remaining_surfel,
		const RemainingSurfelKNN remaining_knn,
		const unsigned* num_remaining_surfels,
		//The output
		LiveSurfelKNNCompactionOutput compaction_output
	) {
		//Query the size at first
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		const auto unfiltered_remaining_size = remaining_surfel.remaining_indicator.Size();
		const auto unfiltered_appended_size = appended_observation.validity_indicator.Size();

		//There is only two types, thus just let it go...
		if(idx < unfiltered_remaining_size) {
			if(remaining_surfel.remaining_indicator[idx] > 0)
			{
				//const auto offset = remaining_indicator_prefixsum[idx] - 1;
				const auto offset = remaining_surfel.remaining_indicator_prefixsum[idx] - 1;
				compaction_output.live_vertex_confid[offset] = remaining_surfel.live_vertex_confid[idx];
				compaction_output.live_normal_radius[offset] = remaining_surfel.live_normal_radius[idx];
				compaction_output.color_time[offset] = remaining_surfel.color_time[idx];
				compaction_output.surfel_knn[offset] = remaining_knn.surfel_knn[idx];
				compaction_output.surfel_knnweight[offset] = remaining_knn.surfel_knn_weight[idx];
			}
		}
		else if(idx >= unfiltered_remaining_size && idx < (unfiltered_remaining_size + unfiltered_appended_size)){
			const auto append_idx = idx - remaining_surfel.remaining_indicator.Size();
			if(appended_observation.validity_indicator[append_idx] > 0)
			{
				const auto offset = appended_observation.validity_indicator_prefixsum[append_idx] + (*num_remaining_surfels) - 1;
				compaction_output.live_vertex_confid[offset] = appended_observation.surfel_vertex_confid[append_idx];
				compaction_output.live_normal_radius[offset] = appended_observation.surfel_normal_radius[append_idx];
				compaction_output.color_time[offset] = appended_observation.surfel_color_time[append_idx];
				compaction_output.surfel_knn[offset] = appended_observation.surfel_knn[append_idx];
				compaction_output.surfel_knnweight[offset] = appended_observation.surfel_knn_weight[append_idx];
			}
		}
	}

	//The method and kernel for compaction of the geometry
	struct ReinitCompactionOutput {
		float4* reference_vertex_confid;
		float4* reference_normal_radius;
		float4* live_vertex_confid;
		float4* live_normal_radius;
		float4* color_time;
	};

	__global__ void compactReinitSurfelKernel(
		const ReinitAppendedObservationSurfel appended_observation,
		const RemainingLiveSurfel remaining_surfel,
		const unsigned* num_remaining_surfels,
		unsigned image_cols,
		ReinitCompactionOutput compaction_output,
		mat34 camera2world
	) {
		//Query the size at first
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		const auto unfiltered_remaining_size = remaining_surfel.remaining_indicator.Size();
		const auto unfiltered_appended_size = appended_observation.validity_indicator.Size();

		//There is only two types, thus just let it go...
		if(idx < unfiltered_remaining_size) {
			if(remaining_surfel.remaining_indicator[idx] > 0)
			{
				//const auto offset = remaining_indicator_prefixsum[idx] - 1;
				const auto offset = remaining_surfel.remaining_indicator_prefixsum[idx] - 1;

				//The vertex
				const float4 vertex_confid = remaining_surfel.live_vertex_confid[idx];
				compaction_output.reference_vertex_confid[offset] = vertex_confid;
				compaction_output.live_vertex_confid[offset] = vertex_confid;

				//The normal
				const float4 normal_radius = remaining_surfel.live_normal_radius[idx];
				compaction_output.reference_normal_radius[offset] = normal_radius;
				compaction_output.live_normal_radius[offset] = normal_radius;

				//The color
				compaction_output.color_time[offset] = remaining_surfel.color_time[idx];
			}
		}
		else if(idx >= unfiltered_remaining_size && idx < (unfiltered_remaining_size + unfiltered_appended_size)){
			const auto append_idx = idx - remaining_surfel.remaining_indicator.Size();
			if(appended_observation.validity_indicator[append_idx] > 0)
			{
				const auto offset = appended_observation.validity_indicator_prefixsum[append_idx] + (*num_remaining_surfels) - 1;

				//The position of this pixel
				const auto x = append_idx % image_cols;
				const auto y = append_idx / image_cols;

				//Query the data
				const float4 vertex_confid = tex2D<float4>(appended_observation.depth_vertex_confid_map, x, y);
				const float3 world_vertex = camera2world.rot * vertex_confid + camera2world.trans;
				compaction_output.reference_vertex_confid[offset] = make_float4(world_vertex.x, world_vertex.y, world_vertex.z, vertex_confid.w);
				compaction_output.live_vertex_confid[offset] = make_float4(world_vertex.x, world_vertex.y, world_vertex.z, vertex_confid.w);

				const float4 normal_radius = tex2D<float4>(appended_observation.depth_normal_radius_map, x, y);
				const float3 world_normal = camera2world.rot * normal_radius;
				compaction_output.reference_normal_radius[offset] = make_float4(world_normal.x, world_normal.y, world_normal.z, normal_radius.w);
				compaction_output.live_normal_radius[offset] = make_float4(world_normal.x, world_normal.y, world_normal.z, normal_radius.w);

				const float4 color_time = tex2D<float4>(appended_observation.observation_color_time_map, x, y);
				compaction_output.color_time[offset] = color_time;
			}
		}
	}

} // device
} // surfelwarp


void surfelwarp::DoubleBufferCompactor::PerformCompactionGeometryKNNSync(
	unsigned& num_valid_remaining_surfels,
	unsigned& num_valid_append_surfels,
	cudaStream_t stream
) {
	//The number of remaining surfel is the last element of inclusive sum
	const unsigned* num_remaining_surfel_dev = m_remaining_surfel.remaining_indicator_prefixsum + m_remaining_surfel.remaining_indicator.Size() - 1;
	
	//Construct the output
	device::LiveSurfelKNNCompactionOutput compaction_output;
	compaction_output.live_vertex_confid = m_compact_to_geometry->m_live_vertex_confid.Ptr();
	compaction_output.live_normal_radius = m_compact_to_geometry->m_live_normal_radius.Ptr();
	compaction_output.color_time = m_compact_to_geometry->m_color_time.Ptr();
	compaction_output.surfel_knn = m_compact_to_geometry->m_surfel_knn.Ptr();
	compaction_output.surfel_knnweight = m_compact_to_geometry->m_surfel_knn_weight.Ptr(); 
	
	//Seems ready for compaction
	const auto unfiltered_remaining_surfel_size = m_remaining_surfel.remaining_indicator.Size();
	const auto unfiltered_appended_surfel_size = m_appended_surfel_knn.validity_indicator.Size();
	dim3 blk(256);
	dim3 grid(divUp(unfiltered_appended_surfel_size + unfiltered_remaining_surfel_size, blk.x));
	device::compactRemainingAndAppendedSurfelKNNKernel<<<grid, blk, 0, stream>>>(
		m_appended_surfel_knn,
		m_remaining_surfel, m_remaining_knn,
		num_remaining_surfel_dev,
		compaction_output
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Sync and query the size
	cudaSafeCall(cudaMemcpyAsync(&num_valid_remaining_surfels, num_remaining_surfel_dev, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
	if(unfiltered_appended_surfel_size > 0) {
		cudaSafeCall(cudaMemcpyAsync(
			&num_valid_append_surfels,
			m_appended_surfel_knn.validity_indicator_prefixsum + m_appended_surfel_knn.validity_indicator.Size() - 1,
			sizeof(unsigned),
			cudaMemcpyDeviceToHost,
			stream
		));
	} else {
		num_valid_append_surfels = 0;
	}
	cudaSafeCall(cudaStreamSynchronize(stream));
	
	//And set the size of output geometry
	m_compact_to_geometry->ResizeValidSurfelArrays(num_valid_append_surfels + num_valid_remaining_surfels);
	
	//Debug code
	//LOG(INFO) << "The number of remaining surfel is " << num_valid_remaining_surfels << " while the size of original surfel is " << m_remaining_surfel.remaining_indicator.Size();
	//LOG(INFO) << "The number of appended surfel is " << num_valid_append_surfels;
}


void surfelwarp::DoubleBufferCompactor::PerformComapctionGeometryOnlySync(
	unsigned & num_valid_remaining_surfels,
	unsigned & num_valid_append_surfels,
	const mat34& camera2world,
	cudaStream_t stream
) {
	//The number of remaining surfel is the last element of inclusive sum
	const unsigned* num_remaining_surfel_dev = m_remaining_surfel.remaining_indicator_prefixsum + m_remaining_surfel.remaining_indicator.Size() - 1;
	
	//Construct the output
	device::ReinitCompactionOutput compaction_output;
	SURFELWARP_CHECK(!(m_compact_to_geometry == nullptr));
	compaction_output.reference_vertex_confid = m_compact_to_geometry->m_reference_vertex_confid.Ptr();
	compaction_output.reference_normal_radius = m_compact_to_geometry->m_reference_normal_radius.Ptr();
	compaction_output.live_vertex_confid = m_compact_to_geometry->m_live_vertex_confid.Ptr();
	compaction_output.live_normal_radius = m_compact_to_geometry->m_live_normal_radius.Ptr();
	compaction_output.color_time = m_compact_to_geometry->m_color_time.Ptr();
	
	//Seems ready for compaction
	const auto unfiltered_remaining_surfel_size = m_remaining_surfel.remaining_indicator.Size();
	const auto unfiltered_appended_surfel_size = m_reinit_append_surfel.validity_indicator.Size();
	dim3 blk(256);
	dim3 grid(divUp(unfiltered_appended_surfel_size + unfiltered_remaining_surfel_size, blk.x));
	device::compactReinitSurfelKernel<<<grid, blk, 0, stream>>>(
		m_reinit_append_surfel,
		m_remaining_surfel,
		num_remaining_surfel_dev,
		m_image_cols,
		compaction_output,
		camera2world
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Sync and query the size
	cudaSafeCall(cudaMemcpyAsync(&num_valid_remaining_surfels, num_remaining_surfel_dev, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
	cudaSafeCall(cudaMemcpyAsync(
		&num_valid_append_surfels,
		m_reinit_append_surfel.validity_indicator_prefixsum + m_reinit_append_surfel.validity_indicator.Size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream
	));
	cudaSafeCall(cudaStreamSynchronize(stream));
	
	//And set the size of output geometry
	m_compact_to_geometry->ResizeValidSurfelArrays(num_valid_append_surfels + num_valid_remaining_surfels);
	
	//Debug code
	//LOG(INFO) << "The number of remaining surfel is " << num_valid_remaining_surfels << " while the size of original surfel is " << m_remaining_surfel.remaining_indicator.Size();
	//LOG(INFO) << "The number of appended surfel is " << num_valid_append_surfels;
}