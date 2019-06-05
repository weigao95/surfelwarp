#include "common/global_configs.h"
#include "core/geometry/FusionRemainingSurfelMarker.h"
#include "core/surfel_format.h"


#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	struct RemainingSurfelMarkerDevice {
		// Some constants defined as enum
		enum {
			scale_factor = d_fusion_map_scale,
			window_halfsize = scale_factor * 2,
			front_threshold = scale_factor * scale_factor * 3,
		};

		//The rendered fusion maps
		struct {
			cudaTextureObject_t vertex_confid_map;
			cudaTextureObject_t normal_radius_map;
			cudaTextureObject_t index_map;
			cudaTextureObject_t color_time_map;
		} fusion_maps;
		
		//The geometry model input
		struct {
			DeviceArrayView<float4> vertex_confid;
			const float4* normal_radius;
			const float4* color_time;
		} live_geometry;

		//The remainin surfel indicator from the fuser
		mutable unsigned* remaining_surfel;

		//the camera and time information
		mat34 world2camera;
		float current_time;

		//The global information
		Intrinsic intrinsic;

		__device__ __forceinline__ void processMarking() const {
			const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
			if(idx >= live_geometry.vertex_confid.Size()) return;

			const float4 surfel_vertex_confid = live_geometry.vertex_confid[idx];
			const float4 surfel_normal_radius = live_geometry.normal_radius[idx];
			const float4 surfel_color_time = live_geometry.color_time[idx];
			//Transfer to camera space
			const float3 vertex = world2camera.rot * surfel_vertex_confid + world2camera.trans;

			//Project to camera image
			const int x = __float2int_rn(((vertex.x / (vertex.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x);
			const int y = __float2int_rn(((vertex.y / (vertex.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y);

			//The corrdinate on ***_map
			const int map_x_center = scale_factor * x; // +windows_halfsize;
			const int map_y_center = scale_factor * y; // +windows_halfsize;
			int front_counter = 0;

			//Window search
			for(auto map_y = map_y_center - window_halfsize; map_y < map_y_center + window_halfsize; map_y++) {
				for(auto map_x = map_x_center - window_halfsize; map_x < map_x_center + window_halfsize; map_x++) {
					const float4 map_vertex_confid = tex2D<float4>(fusion_maps.vertex_confid_map, map_x, map_y);
					const float4 map_normal_radius = tex2D<float4>(fusion_maps.normal_radius_map, map_x, map_y);
					const auto index = tex2D<unsigned>(fusion_maps.index_map, map_x, map_y);
					if(index != 0xFFFFFFFF)
					{
						const auto dot_value =dotxyz(surfel_normal_radius, map_normal_radius);
						const float3 diff_camera = world2camera.rot * (surfel_vertex_confid - map_vertex_confid);
						if(diff_camera.z >= 0 && diff_camera.z <= 3 * 0.001 && dot_value >= 0.8) {
							front_counter++;
						}
					}
				}
			}

			//The global counter
			unsigned keep_indicator = 1;

			//Check the number of front surfels
			if(front_counter > front_threshold) keep_indicator = 0;

			//Check the initialize time
			if(surfel_vertex_confid.w < 10.0f && (current_time - initialization_time(surfel_color_time)) > 30.0f) keep_indicator = 0;

			//Write to output
			if(keep_indicator == 1 && remaining_surfel[idx] == 0) {
				remaining_surfel[idx] = 1;
			}
		}
		
	};


	__global__ void markRemainingSurfelKernel(
		const RemainingSurfelMarkerDevice marker
	) {
		marker.processMarking();
	}


} // device
} // surfelwarp


void surfelwarp::FusionRemainingSurfelMarker::UpdateRemainingSurfelIndicator(cudaStream_t stream) {
	//Construct the marker
	device::RemainingSurfelMarkerDevice marker;
	
	marker.fusion_maps.vertex_confid_map = m_fusion_maps.vertex_confid_map;
	marker.fusion_maps.normal_radius_map = m_fusion_maps.normal_radius_map;
	marker.fusion_maps.index_map = m_fusion_maps.index_map;
	marker.fusion_maps.color_time_map = m_fusion_maps.color_time_map;
	
	marker.live_geometry.vertex_confid = m_live_geometry.vertex_confid;
	marker.live_geometry.normal_radius = m_live_geometry.normal_radius.RawPtr();
	marker.live_geometry.color_time = m_live_geometry.color_time.RawPtr();
	
	marker.remaining_surfel = m_remaining_surfel_indicator.RawPtr();
	marker.world2camera = m_world2camera;
	marker.current_time = m_current_time;
	marker.intrinsic = m_intrinsic;
	
	dim3 blk(256);
	dim3 grid(divUp(m_live_geometry.vertex_confid.Size(), blk.x));
	device::markRemainingSurfelKernel<<<grid, blk, 0, stream>>>(marker);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::FusionRemainingSurfelMarker::RemainingSurfelIndicatorPrefixSum(cudaStream_t stream) {
	m_remaining_indicator_prefixsum.InclusiveSum(m_remaining_surfel_indicator.ArrayView(), stream);
}

surfelwarp::DeviceArrayView<unsigned int>
surfelwarp::FusionRemainingSurfelMarker::GetRemainingSurfelIndicatorPrefixsum() const {
	const auto& prefixsum_array = m_remaining_indicator_prefixsum.valid_prefixsum_array;
	SURFELWARP_CHECK(m_remaining_surfel_indicator.Size() == prefixsum_array.size());
	return DeviceArrayView<unsigned>(prefixsum_array);
}

