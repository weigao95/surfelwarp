#include "common/global_configs.h"
#include "core/geometry/SurfelFusionHandler.h"
#include "core/geometry/surfel_fusion_dev.cuh"
#include "math/vector_ops.hpp"
#include "math/device_mat.h"

#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	struct FusionAndMarkAppendedObservationSurfelDevice {
		// Some constants defined as enum
		enum {
			scale_factor = d_fusion_map_scale,
			fuse_window_halfsize = scale_factor >> 1,
			count_model_halfsize = 2 * scale_factor /*>> 1 */,
			append_window_halfsize = scale_factor,
			search_window_halfsize = scale_factor,
		};


		//The observation
		struct {
			cudaTextureObject_t vertex_time_map;
			cudaTextureObject_t normal_radius_map;
			cudaTextureObject_t color_time_map;
		} observation_maps;

		//The rendered maps
		struct {
			cudaTextureObject_t vertex_map;
			cudaTextureObject_t normal_map;
			cudaTextureObject_t color_time_map;
			cudaTextureObject_t index_map;
		} render_maps;

		//The written array
		struct {
			float4* vertex_confidence;
			float4* normal_radius;
			float4* color_time;
			unsigned* fused_indicator;
		} geometry_arrays;

		//The shared datas
		unsigned short image_rows, image_cols;
		float current_time;
		
		__host__ __device__ __forceinline__ bool checkViewDirection(
			const float4& depth_vertex, const float4& depth_normal
		) const {
			const float3 view_direction = - normalized(make_float3(depth_vertex.x, depth_vertex.y, depth_vertex.z));
			const float3 normal = normalized(make_float3(depth_normal.x, depth_normal.y, depth_normal.z));
			return dot(view_direction, normal) > 0.4f;
		}
		

		//The actual processing interface
		__device__ __forceinline__ void processIndicator(const mat34& world2camera, unsigned* appending_indicator) const {
			const int x = threadIdx.x + blockDim.x * blockIdx.x;
			const int y = threadIdx.y + blockDim.y * blockIdx.y;
			const auto offset = y * image_cols + x;
			if(x < search_window_halfsize || x >= image_cols - search_window_halfsize 
				|| y < search_window_halfsize || y >= image_rows - search_window_halfsize) {
				//Write to indicator before exit
				appending_indicator[offset] = 0;
				return;
			}

			//Load the data
			const float4 depth_vertex_confid = tex2D<float4>(observation_maps.vertex_time_map, x, y);
			const float4 depth_normal_radius = tex2D<float4>(observation_maps.normal_radius_map, x, y);
			const float4 image_color_time = tex2D<float4>(observation_maps.color_time_map, x, y);
			if(is_zero_vertex(depth_vertex_confid)) return;

			//The windows search state
			const int map_x_center = scale_factor * x;
			const int map_y_center = scale_factor * y;

			//The window search iteration variables
			SurfelFusionWindowState fusion_state;
			unsigned model_count = 0;
			SurfelAppendingWindowState append_state;

			//The row search loop
			for(int dy = - search_window_halfsize; dy < search_window_halfsize; dy++) {
				for(int dx = - search_window_halfsize; dx < search_window_halfsize; dx++) {
					//The actual position of in the rendered map
					const int map_y = dy + map_y_center;
					const int map_x = dx + map_x_center;

					const auto index = tex2D<unsigned>(render_maps.index_map, map_x, map_y);
					if(index != 0xFFFFFFFF)
					{
						//Load the model vertex
						const float4 model_world_v4 = tex2D<float4>(render_maps.vertex_map, map_x, map_y);
						const float4 model_world_n4 = tex2D<float4>(render_maps.normal_map, map_x, map_y);
						
						//Transform it
						const float3 model_camera_v3 = world2camera.rot * model_world_v4 + world2camera.trans;
						const float3 model_camera_n3 = world2camera.rot * model_world_n4;
						
						//Some attributes commonly used for checking
						const float dot_value = dotxyz(model_camera_n3, depth_normal_radius);
						const float diff_z = fabsf(model_camera_v3.z - depth_vertex_confid.z);
						const float confidence = model_world_v4.w;
						const float z_dist = model_camera_v3.z;
						const float dist_square = squared_distance(model_camera_v3, depth_vertex_confid);

						//First check for fusion
						if(dx >= -fuse_window_halfsize && dy >= -fuse_window_halfsize && dx < fuse_window_halfsize && dy < fuse_window_halfsize)
						{
							if(dot_value >= 0.8f && diff_z <= 3 * 0.001f) { // Update it
								fusion_state.Update(confidence, z_dist, map_x, map_y);
							}
						}

						//Next check for count the model
						if(dx >= -count_model_halfsize && dy >= -count_model_halfsize && dx < count_model_halfsize && dy < count_model_halfsize) {
							if(dot_value > 0.3f)
								model_count++;
						}

						//Finally for appending
						{
							//if(dot_value >= 0.8f && diff_z <= 3 * 0.001f) { // Update it
							//	append_state.Update(confidence, z_dist);
							//}
							if(dot_value >= 0.8f && dist_square <= (2 * 0.001f) * (2 * 0.001f)) { // Update it
								append_state.Update(confidence, z_dist);
							}
						}

					} // There is a surfel here
				} // x iteration loop
			} // y iteration loop
			
			//For appending, as in reinit should mark all depth surfels
			unsigned pixel_indicator = 0;
			if(append_state.best_confid < -0.01
			   && model_count == 0
			   && checkViewDirection(depth_vertex_confid, depth_normal_radius)
				) {
				pixel_indicator = 1;
			}
			appending_indicator[offset] = pixel_indicator;

			//For fusion
			if(fusion_state.best_confid > 0) {
				float4 model_vertex_confid = tex2D<float4>(render_maps.vertex_map, fusion_state.best_map_x, fusion_state.best_map_y);
				float4 model_normal_radius = tex2D<float4>(render_maps.normal_map, fusion_state.best_map_x, fusion_state.best_map_y);
				float4 model_color_time = tex2D<float4>(render_maps.color_time_map, fusion_state.best_map_x, fusion_state.best_map_y);
				const unsigned index = tex2D<unsigned>(render_maps.index_map, fusion_state.best_map_x, fusion_state.best_map_y);
				fuse_surfel(
					depth_vertex_confid, depth_normal_radius, image_color_time, 
					world2camera, current_time,
					model_vertex_confid, model_normal_radius, model_color_time
				);

				//Write it
				geometry_arrays.vertex_confidence[index] = model_vertex_confid;
				geometry_arrays.normal_radius[index] = model_normal_radius;
				geometry_arrays.color_time[index] = model_color_time;
				geometry_arrays.fused_indicator[index] = 1;
			}
		}

		__device__ __forceinline__ void processAtomic(const mat34& world2camera, unsigned* appending_offset, ushort2* appended_pixels) const {
			const int x = threadIdx.x + blockDim.x * blockIdx.x;
			const int y = threadIdx.y + blockDim.y * blockIdx.y;
			if(x < search_window_halfsize || x >= image_cols - search_window_halfsize 
				|| y < search_window_halfsize || y >= image_rows - search_window_halfsize) return;

			//Load the data
			const float4 depth_vertex_confid = tex2D<float4>(observation_maps.vertex_time_map, x, y);
			const float4 depth_normal_radius = tex2D<float4>(observation_maps.normal_radius_map, x, y);
			const float4 image_color_time = tex2D<float4>(observation_maps.color_time_map, x, y);
			if(is_zero_vertex(depth_vertex_confid)) return;

			//The windows search state
			const int map_x_center = scale_factor * x;
			const int map_y_center = scale_factor * y;

			//The window search iteration variables
			unsigned model_count = 0;
			SurfelFusionWindowState fusion_state;
			SurfelAppendingWindowState append_state;

			//The row search loop
			for(int dy = - search_window_halfsize; dy < search_window_halfsize; dy++) {
				for(int dx = - search_window_halfsize; dx < search_window_halfsize; dx++) {
					//The actual position of in the rendered map
					const int map_y = dy + map_y_center;
					const int map_x = dx + map_x_center;

					const auto index = tex2D<unsigned>(render_maps.index_map, map_x, map_y);
					if(index != 0xFFFFFFFF)
					{
						//Load the model vertex
						const float4 model_world_v4 = tex2D<float4>(render_maps.vertex_map, map_x, map_y);
						const float4 model_world_n4 = tex2D<float4>(render_maps.normal_map, map_x, map_y);
						
						//Transform it
						const float3 model_camera_v3 = world2camera.rot * model_world_v4 + world2camera.trans;
						const float3 model_camera_n3 = world2camera.rot * model_world_n4;
						
						//Some attributes commonly used for checking
						const float dot_value = dotxyz(model_camera_n3, depth_normal_radius);
						const float diff_z = fabsf(model_camera_v3.z - depth_vertex_confid.z);
						const float confidence = model_world_v4.w;
						const float z_dist = model_camera_v3.z;
						const float dist_square = squared_distance(model_camera_v3, depth_vertex_confid);

						//First check for fusion
						if(dx >= -fuse_window_halfsize && dy >= -fuse_window_halfsize && dx < fuse_window_halfsize && dy < fuse_window_halfsize)
						{
							if(dot_value >= 0.8f && diff_z <= 3 * 0.001f) { // Update it
								fusion_state.Update(confidence, z_dist, map_x, map_y);
							}
						}

						//Next check for count the model
						if(dx >= -count_model_halfsize && dy >= -count_model_halfsize && dx < count_model_halfsize && dy < count_model_halfsize) {
								model_count++;
						}

						//Finally for appending
						{
							if(dot_value >= 0.8f && dist_square <= (3 * 0.001f) * (3 * 0.001f)) { // Update it
								append_state.Update(confidence, z_dist);
							}
						}

					} // There is a surfel here
				} // x iteration loop
			} // y iteration loop

			//For fusion
			if(fusion_state.best_confid > 0) {
				float4 model_vertex_confid = tex2D<float4>(render_maps.vertex_map, fusion_state.best_map_x, fusion_state.best_map_y);
				float4 model_normal_radius = tex2D<float4>(render_maps.normal_map, fusion_state.best_map_x, fusion_state.best_map_y);
				float4 model_color_time = tex2D<float4>(render_maps.color_time_map, fusion_state.best_map_x, fusion_state.best_map_y);
				const unsigned index = tex2D<unsigned>(render_maps.index_map, fusion_state.best_map_x, fusion_state.best_map_y);
				fuse_surfel(
					depth_vertex_confid, depth_normal_radius, image_color_time, 
					world2camera, current_time,
					model_vertex_confid, model_normal_radius, model_color_time
				);


				//Write it
				geometry_arrays.vertex_confidence[index] = model_vertex_confid;
				geometry_arrays.normal_radius[index] = model_normal_radius;
				geometry_arrays.color_time[index] = model_color_time;
				geometry_arrays.fused_indicator[index] = 1;
			}
			

			//Check the view direction, and using atomic operation for appending
			if(append_state.best_confid < -0.01
			   && model_count == 0
			   && checkViewDirection(depth_vertex_confid, depth_normal_radius)
				) {
				const auto offset = atomicAdd(appending_offset, 1);
				appended_pixels[offset] = make_ushort2(x, y);
			}	
		}


		//The fusion processor for re-initialize
		__device__ __forceinline__ void processFusionReinit(const mat34& world2camera, unsigned* appending_indicator) const {
			const int x = threadIdx.x + blockDim.x * blockIdx.x;
			const int y = threadIdx.y + blockDim.y * blockIdx.y;
			const auto offset = y * image_cols + x;
			if(x < search_window_halfsize || x >= image_cols - search_window_halfsize 
				|| y < search_window_halfsize || y >= image_rows - search_window_halfsize) {
				//Write to indicator before exit
				appending_indicator[offset] = 0;
				return;
			}

			//Load the data
			const float4 depth_vertex_confid = tex2D<float4>(observation_maps.vertex_time_map, x, y);
			const float4 depth_normal_radius = tex2D<float4>(observation_maps.normal_radius_map, x, y);
			const float4 image_color_time = tex2D<float4>(observation_maps.color_time_map, x, y);
			if(is_zero_vertex(depth_vertex_confid)) return;

			//The windows search state
			const int map_x_center = scale_factor * x;
			const int map_y_center = scale_factor * y;

			//The window search iteration variables
			SurfelFusionWindowState fusion_state;

			//The row search loop
			for(int dy = - fuse_window_halfsize; dy < fuse_window_halfsize; dy++) {
				for(int dx = - fuse_window_halfsize; dx < fuse_window_halfsize; dx++) {
					//The actual position of in the rendered map
					const int map_y = dy + map_y_center;
					const int map_x = dx + map_x_center;

					const auto index = tex2D<unsigned>(render_maps.index_map, map_x, map_y);
					if(index != 0xFFFFFFFF)
					{
						//Load the model vertex
						const float4 model_world_v4 = tex2D<float4>(render_maps.vertex_map, map_x, map_y);
						const float4 model_world_n4 = tex2D<float4>(render_maps.normal_map, map_x, map_y);
						
						//Transform it
						const float3 model_camera_v3 = world2camera.rot * model_world_v4 + world2camera.trans;
						const float3 model_camera_n3 = world2camera.rot * model_world_n4;
						
						//Some attributes commonly used for checking
						const float dot_value = dotxyz(model_camera_n3, depth_normal_radius);
						const float diff_z = fabsf(model_camera_v3.z - depth_vertex_confid.z);
						const float confidence = model_world_v4.w;
						const float z_dist = model_camera_v3.z;
						//const float dist_square = squared_distance(model_camera_v3, depth_vertex_confid);

						//First check for fusion
						if(dot_value >= 0.9f && diff_z <= 2 * 0.001f) { // Update it
							fusion_state.Update(confidence, z_dist, map_x, map_y);
						}
					} // There is a surfel here
				} // x iteration loop
			} // y iteration loop
			
			//For appending, as in reinit should mark all depth surfels
			unsigned pixel_indicator = 0;
			if(fusion_state.best_confid < -0.01) {
				pixel_indicator = 1;
			}
			appending_indicator[offset] = pixel_indicator;

			//For fusion
			if(fusion_state.best_confid > 0) {
				float4 model_vertex_confid = tex2D<float4>(render_maps.vertex_map, fusion_state.best_map_x, fusion_state.best_map_y);
				float4 model_normal_radius = tex2D<float4>(render_maps.normal_map, fusion_state.best_map_x, fusion_state.best_map_y);
				float4 model_color_time = tex2D<float4>(render_maps.color_time_map, fusion_state.best_map_x, fusion_state.best_map_y);
				const unsigned index = tex2D<unsigned>(render_maps.index_map, fusion_state.best_map_x, fusion_state.best_map_y);
				fuse_surfel_replace_color(
					depth_vertex_confid, depth_normal_radius, image_color_time, 
					world2camera, current_time,
					model_vertex_confid, model_normal_radius, model_color_time
				);

				//Write it
				geometry_arrays.vertex_confidence[index] = model_vertex_confid;
				geometry_arrays.normal_radius[index] = model_normal_radius;
				geometry_arrays.color_time[index] = model_color_time;
				geometry_arrays.fused_indicator[index] = 1;
			}
		}


	};


	__global__ void fuseAndMarkAppendedObservationSurfelsKernel(
		const FusionAndMarkAppendedObservationSurfelDevice fuser,
		mat34 world2camera,
		unsigned* appended_pixel
	) {
		fuser.processIndicator(world2camera, appended_pixel);
	}

	__global__ void fusionAndMarkAppendObservationAtomicKernel(
		const FusionAndMarkAppendedObservationSurfelDevice fuser,
		mat34 world2camera,
		unsigned* append_offset,
		ushort2* appended_pixel
	) {
		fuser.processAtomic(world2camera, append_offset, appended_pixel);
	}

	__global__ void fuseAndMarkAppendedObservationSurfelReinitKernel(
		const FusionAndMarkAppendedObservationSurfelDevice fuser,
		mat34 world2camera,
		unsigned* appended_pixel
	) {
		fuser.processFusionReinit(world2camera, appended_pixel);
	}


	__global__ void compactIndicatorToPixelKernel(
		const unsigned* candidate_pixel_indicator,
		const unsigned* prefixsum_indicator,
		unsigned img_cols,
		ushort2* compacted_pixels
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(candidate_pixel_indicator[idx] > 0) {
			const auto offset = prefixsum_indicator[idx] - 1;
			const unsigned short x = idx % img_cols;
			const unsigned short y = idx / img_cols;
			compacted_pixels[offset] = make_ushort2(x, y);
		}
	}

} // device
} // surfelwarp


void surfelwarp::SurfelFusionHandler::prepareFuserArguments(void *fuser_ptr) {
	//Recovery the fuser arguments
	auto& fuser = *((device::FusionAndMarkAppendedObservationSurfelDevice*)fuser_ptr);
	
	//The observation maps
	fuser.observation_maps.vertex_time_map = m_observation.vertex_config_map;
	fuser.observation_maps.normal_radius_map = m_observation.normal_radius_map;
	fuser.observation_maps.color_time_map = m_observation.color_time_map;
	
	//The rendered maps
	fuser.render_maps.vertex_map = m_fusion_maps.warp_vertex_map;
	fuser.render_maps.normal_map = m_fusion_maps.warp_normal_map;
	fuser.render_maps.index_map = m_fusion_maps.index_map;
	fuser.render_maps.color_time_map = m_fusion_maps.color_time_map;
	
	//The written array
	fuser.geometry_arrays.vertex_confidence = m_fusion_geometry.live_vertex_confid.RawPtr();
	fuser.geometry_arrays.normal_radius = m_fusion_geometry.live_normal_radius.RawPtr();
	fuser.geometry_arrays.color_time = m_fusion_geometry.color_time.RawPtr();
	fuser.geometry_arrays.fused_indicator = m_remaining_surfel_indicator.Ptr();
	
	//Other attributes
	fuser.current_time = m_current_time;
	fuser.image_cols = m_image_cols;
	fuser.image_rows = m_image_rows;
}

void surfelwarp::SurfelFusionHandler::processFusionAppendCompaction(cudaStream_t stream)
{
	//Resize the array
	const auto num_surfels = m_fusion_geometry.live_vertex_confid.Size();
	m_remaining_surfel_indicator.ResizeArrayOrException(num_surfels);
	
	//Construct the fuser
	device::FusionAndMarkAppendedObservationSurfelDevice fuser;
	prepareFuserArguments((void*)&fuser);
	
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_cols, blk.x), divUp(m_image_rows, blk.y));
	device::fuseAndMarkAppendedObservationSurfelsKernel<<<grid, blk, 0, stream>>>(
		fuser, 
		m_world2camera, 
		m_appended_depth_surfel_indicator.ptr()
	);
}

void surfelwarp::SurfelFusionHandler::processFusionReinit(cudaStream_t stream)
{
	//Resize the array
	const auto num_surfels = m_fusion_geometry.live_vertex_confid.Size();
	m_remaining_surfel_indicator.ResizeArrayOrException(num_surfels);
	
	//Construct the fuser
	device::FusionAndMarkAppendedObservationSurfelDevice fuser;
	prepareFuserArguments((void*)&fuser);
	
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_cols, blk.x), divUp(m_image_rows, blk.y));
	device::fuseAndMarkAppendedObservationSurfelReinitKernel<<<grid, blk, 0, stream>>>(
		fuser, 
		m_world2camera, 
		m_appended_depth_surfel_indicator.ptr()
	);
}

void surfelwarp::SurfelFusionHandler::processFusionAppendAtomic(cudaStream_t stream)
{
	//Clear the attributes
	cudaSafeCall(cudaMemsetAsync(m_atomic_appended_pixel_index, 0, sizeof(unsigned), stream));

	//Resize the array
	const auto num_surfels = m_fusion_geometry.live_vertex_confid.Size();
	m_remaining_surfel_indicator.ResizeArrayOrException(num_surfels);
	
	//Construct the fuser
	device::FusionAndMarkAppendedObservationSurfelDevice fuser;
	prepareFuserArguments((void*)&fuser);
	
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_cols, blk.x), divUp(m_image_rows, blk.y));
	device::fusionAndMarkAppendObservationAtomicKernel<<<grid, blk, 0, stream>>>(
		fuser, 
		m_world2camera,
		m_atomic_appended_pixel_index, 
		m_atomic_appended_observation_pixel.Ptr()
	);
}


void surfelwarp::SurfelFusionHandler::compactAppendedIndicator(cudaStream_t stream) {
	m_appended_surfel_indicator_prefixsum.InclusiveSum(m_appended_depth_surfel_indicator, stream);
	
	//Invoke the kernel
	dim3 blk(128);
	dim3 grid(divUp(m_image_cols * m_image_rows, blk.x));
	device::compactIndicatorToPixelKernel<<<grid, blk, 0, stream>>>(
		m_appended_depth_surfel_indicator.ptr(),
		m_appended_surfel_indicator_prefixsum.valid_prefixsum_array.ptr(),
		m_image_cols,
		m_compacted_appended_pixel.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Query the size
	unsigned num_appended_surfel;
	cudaSafeCall(cudaMemcpyAsync(
		&num_appended_surfel,
		m_appended_surfel_indicator_prefixsum.valid_prefixsum_array.ptr() + m_appended_surfel_indicator_prefixsum.valid_prefixsum_array.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream
	));
	
	cudaSafeCall(cudaStreamSynchronize(stream));
	m_compacted_appended_pixel.ResizeArrayOrException(num_appended_surfel);
}

void surfelwarp::SurfelFusionHandler::queryAtomicAppendedPixelSize(cudaStream_t stream) {
	unsigned num_candidate_pixels;
	cudaSafeCall(cudaMemcpyAsync(
		&num_candidate_pixels,
		m_atomic_appended_pixel_index,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream)
	);
	cudaSafeCall(cudaStreamSynchronize(stream));
	m_atomic_appended_observation_pixel.ResizeArrayOrException(num_candidate_pixels);
}