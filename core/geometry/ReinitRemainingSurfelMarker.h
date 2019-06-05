//
// Created by wei on 5/19/18.
//

#pragma once

#include "common/ArrayView.h"
#include "common/ArraySlice.h"
#include "common/macro_utils.h"
#include "common/algorithm_types.h"
#include "common/CameraObservation.h"
#include "math/device_mat.h"
#include "core/render/Renderer.h"
#include "core/warp_solver/solver_types.h"
#include "core/geometry/fusion_types.h"

#include <memory>

namespace surfelwarp {


	class ReinitRemainingSurfelMarker {
	private:
		//The input from outside
		Renderer::FusionMaps m_fusion_maps;

		//With the transform form observation to world
		CameraObservation m_observation;
		mat34 m_world2camera;

		//The geometry as array
		SurfelGeometry::SurfelFusionInput m_surfel_geometry;

		//The fused remaining indicator, where 1 indicates the surfel is fused with some depth image
		DeviceArraySlice<unsigned> m_remaining_surfel_indicator;

		//To project the surfel into image
		Intrinsic m_intrinsic;
	public:
		using Ptr = std::shared_ptr<ReinitRemainingSurfelMarker>;
		ReinitRemainingSurfelMarker();
		~ReinitRemainingSurfelMarker() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(ReinitRemainingSurfelMarker);
		
		void SetInputs(
			const Renderer::FusionMaps& maps,
			const SurfelGeometry::SurfelFusionInput& geometry,
			const CameraObservation& observation,
			float current_time,
			const mat34& world2camera,
			const DeviceArraySlice<unsigned>& remaining_surfel_indicator
		);


		//The processing interface
	private:
		void prepareMarkerArguments(void* raw_marker);
	public:
		void MarkRemainingSurfelObservedOnly(cudaStream_t stream = 0);
		void MarkRemainingSurfelNodeError(const NodeAlignmentError& node_error, float threshold = 0.06f, cudaStream_t stream = 0);
		DeviceArraySlice<unsigned> GetRemainingSurfelIndicator() { return m_remaining_surfel_indicator; }
	};

}