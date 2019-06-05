#include "core/geometry/ReinitRemainingSurfelMarker.h"
#include "core/warp_solver/solver_types.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {


	struct ReinitRemainingMarkerDevice {

		enum {
			window_halfsize = 2,
		};

		//The geometry model input
		struct {
			DeviceArrayView<float4> vertex_confid;
			const float4* normal_radius;
			const float4* color_time;
			const ushort4* surfel_knn;
		} live_geometry;

		//The observation from camera
		struct {
			cudaTextureObject_t vertex_map;
			cudaTextureObject_t normal_map;
			cudaTextureObject_t foreground_mask;
		} camera_observation;

		//The information on camera
		mat34 world2camera;
		Intrinsic intrinsic;

		__device__ __forceinline__ void processMarkingObservedOnly(unsigned* remaining_indicator) const {
			const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
			if(idx >= live_geometry.vertex_confid.Size()) return;

			//Is this surfel fused? If so, must remain
			const auto fused = remaining_indicator[idx];
			if(fused > 0) return;

			const float4 surfel_vertex_confid = live_geometry.vertex_confid[idx];
			const float4 surfel_normal_radius = live_geometry.normal_radius[idx];
			//const float4 surfel_color_time = live_geometry.color_time[idx];
			
			//Transfer to camera space
			const float3 vertex = world2camera.rot * surfel_vertex_confid + world2camera.trans;
			const float3 normal = world2camera.rot * surfel_normal_radius;

			//Project to camera image
			const int x = __float2int_rn(((vertex.x / (vertex.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x);
			const int y = __float2int_rn(((vertex.y / (vertex.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y);

			//Deal with the case where (x, y) is out of the range of the image

			//The flag value
			bool has_corresponded = false;

			//Does this surfel has correspondece on depth image?
			for(auto map_y = y - window_halfsize; map_y < y + window_halfsize; map_y++) {
				for(auto map_x = x - window_halfsize; map_x < x + window_halfsize; map_x++) {
					//Load the depth image
					const float4 depth_vertex = tex2D<float4>(camera_observation.vertex_map, map_x, map_y);
					const float4 depth_normal = tex2D<float4>(camera_observation.normal_map, map_x, map_y);
					
					//Compute various values
					const float normal_dot = dotxyz(normal, depth_normal);
					
					//Check for correspond
					if(squared_distance(vertex, depth_vertex) < 0.003f * 0.003f && normal_dot >= 0.8f)
						has_corresponded = true;
				}
			} // windows search on depth image

			//Check the foreground
			auto foregound = tex2D<unsigned char>(camera_observation.foreground_mask, x, y);

			unsigned remain = 0;
			if(has_corresponded && (foregound > 0))
				remain = 1;

			//Write to output
			remaining_indicator[idx] = remain;
		}


		__device__ __forceinline__ void processMarkingNodeError(
			const NodeAlignmentError& node_error,
			float threshold,
			unsigned* remaining_indicator
		) const {
			const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
			if(idx >= live_geometry.vertex_confid.Size()) return;

			//Is this surfel fused? If so, must remain
			const auto fused = remaining_indicator[idx];
			if(fused > 0) return;

			const float4 surfel_vertex_confid = live_geometry.vertex_confid[idx];
			const float4 surfel_normal_radius = live_geometry.normal_radius[idx];
			//const float4 surfel_color_time = live_geometry.color_time[idx];
			
			//Transfer to camera space
			const float3 vertex = world2camera.rot * surfel_vertex_confid + world2camera.trans;
			//const float3 normal = world2camera.rot * surfel_normal_radius;

			//Project to camera image and check foreground
			const int x = __float2int_rn(((vertex.x / (vertex.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x);
			const int y = __float2int_rn(((vertex.y / (vertex.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y);
			const auto foregound = tex2D<unsigned char>(camera_observation.foreground_mask, x, y);

			//Somehow optimistic
			unsigned remain = foregound > 0 ? 1 : 0;

			//Check the error
			const ushort4 knn_nodes = live_geometry.surfel_knn[idx];
			const unsigned short* knn_nodes_flat = (const unsigned short*)(&knn_nodes);
			for(auto i = 0; i < 4; i++) {
				const auto node = knn_nodes_flat[i];
				const float accumlate_error = node_error.node_accumlated_error[node];
				const float accumlate_weight = node_error.node_accumlate_weight[node];
				if(accumlate_weight * threshold > accumlate_error)
					remain = 0;
			}

			//Write to output
			remaining_indicator[idx] = remain;
		}
	};


	__global__ void markReinitRemainingSurfelObservedOnlyKernel(
		const ReinitRemainingMarkerDevice marker,
		unsigned* remaining_indicator
	) {
		marker.processMarkingObservedOnly(remaining_indicator);
	}


	__global__ void markReinitRemainingSurfelNodeErrorKernel(
		const ReinitRemainingMarkerDevice marker,
		const NodeAlignmentError node_error,
		float threshold,
		unsigned* remaining_indicator
	) {
		marker.processMarkingNodeError(node_error, threshold, remaining_indicator);
	}

} // device
} // surfelwarp

void surfelwarp::ReinitRemainingSurfelMarker::prepareMarkerArguments(void * raw_marker) {
	device::ReinitRemainingMarkerDevice& marker = *((device::ReinitRemainingMarkerDevice*)raw_marker);

	marker.live_geometry.vertex_confid = m_surfel_geometry.live_vertex_confid.ArrayView();
	marker.live_geometry.normal_radius = m_surfel_geometry.live_normal_radius.RawPtr();
	marker.live_geometry.color_time = m_surfel_geometry.color_time.RawPtr();
	marker.live_geometry.surfel_knn = m_surfel_geometry.surfel_knn.RawPtr();
	
	marker.camera_observation.vertex_map = m_observation.vertex_config_map;
	marker.camera_observation.normal_map = m_observation.normal_radius_map;
	marker.camera_observation.foreground_mask = m_observation.foreground_mask;

	marker.world2camera = m_world2camera;
	marker.intrinsic = m_intrinsic;
}


void surfelwarp::ReinitRemainingSurfelMarker::MarkRemainingSurfelObservedOnly(cudaStream_t stream) {
	//Construct the argument
	device::ReinitRemainingMarkerDevice marker;
	prepareMarkerArguments((void*)&marker);

	//Invoke the kernel
	dim3 blk(256);
	dim3 grid(divUp(m_remaining_surfel_indicator.Size(), blk.x));
	device::markReinitRemainingSurfelObservedOnlyKernel<<<grid, blk, 0, stream>>>(
		marker, 
		m_remaining_surfel_indicator.RawPtr()
	);


	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::ReinitRemainingSurfelMarker::MarkRemainingSurfelNodeError(
	const NodeAlignmentError & node_error, 
	float threshold, 
	cudaStream_t stream
) {
	//Construct the argument
	device::ReinitRemainingMarkerDevice marker;
	prepareMarkerArguments((void*)&marker);

	//Invoke the kernel
	dim3 blk(256);
	dim3 grid(divUp(m_remaining_surfel_indicator.Size(), blk.x));
	device::markReinitRemainingSurfelNodeErrorKernel<<<grid, blk, 0, stream>>>(
		marker,
		node_error, 
		threshold,
		m_remaining_surfel_indicator.RawPtr()
	);


	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}