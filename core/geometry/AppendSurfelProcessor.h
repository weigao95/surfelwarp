//
// Created by wei on 5/4/18.
//

#pragma once

#include "common/algorithm_types.h"
#include "common/CameraObservation.h"
#include "core/WarpField.h"
#include "core/geometry/fusion_types.h"
#include <memory>

namespace surfelwarp {
	
	
	class AppendSurfelProcessor {
	private:
		//The observation from depth image
		struct {
			cudaTextureObject_t vertex_confid_map;
			cudaTextureObject_t normal_radius_map;
			cudaTextureObject_t color_time_map;
		} m_observation;
		mat34 m_camera2world;
		
		//The input from warp field
		WarpField::LiveGeometryUpdaterInput m_warpfield_input;
		KNNSearch::Ptr m_live_node_skinner;
		
		//The input indicator for pixel or binary indicator?
		DeviceArrayView<ushort2> m_surfel_candidate_pixel;
		
	public:
		using Ptr = std::shared_ptr<AppendSurfelProcessor>;
		AppendSurfelProcessor();
		~AppendSurfelProcessor();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(AppendSurfelProcessor);
		
		//The input interface
		void SetInputs(
			const CameraObservation& observation,
			const mat34& camera2world,
			const WarpField::LiveGeometryUpdaterInput& warpfield_input,
			const KNNSearch::Ptr& live_node_skinner,
			const DeviceArrayView<ushort2>& pixel_coordinate
		);


		/* The surfel used for compute finite difference. This
		 * version use only xyz component
		 */
	private:
		DeviceBufferArray<float4> m_surfel_vertex_confid;
		DeviceBufferArray<float4> m_surfel_normal_radius;
		DeviceBufferArray<float4> m_surfel_color_time;
		DeviceBufferArray<float4> m_candidate_vertex_finite_diff;
		static constexpr const int kNumFiniteDiffVertex = 4;
		static constexpr const float kFiniteDiffStep = 5e-3f; // 5 [mm]
	public:
		void BuildSurfelAndFiniteDiffVertex(cudaStream_t stream = 0);
		
		
		/* Perform skinning of the vertex using live vertex
		 */
	private:
		DeviceBufferArray<ushort4> m_candidate_vertex_finitediff_knn;
		DeviceBufferArray<float4> m_candidate_vertex_finitediff_knnweight;
	public:
		void SkinningFiniteDifferenceVertex(cudaStream_t stream = 0);
		
		
		/* The buffer and method to perform filtering
		 */
	private:
		DeviceBufferArray<unsigned> m_candidate_surfel_validity_indicator;
		DeviceBufferArray<ushort4> m_surfel_knn;
		DeviceBufferArray<float4> m_surfel_knn_weight;
		
		//Do a prefix sum for the validity indicator
		PrefixSum m_candidate_surfel_validity_prefixsum;
	public:
		void FilterCandidateSurfels(cudaStream_t stream = 0);
		
		
		/* The accessing interface
		 */
		AppendedObservationSurfelKNN GetAppendedObservationSurfel() const;
	};
	
	
}
