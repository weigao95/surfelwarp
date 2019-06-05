//
// Created by wei on 5/19/18.
//

#pragma once

#include "core/SurfelGeometry.h"
#include "core/warp_solver/solver_types.h"
#include "core/geometry/SurfelFusionHandler.h"
#include "core/geometry/ReinitRemainingSurfelMarker.h"
#include "core/geometry/DoubleBufferCompactor.h"

#include <memory>

namespace surfelwarp {
	
	/* The geometry reinitialization processor, as its name, 
	 * takes input from current geometry (both array and rendered map) and
	 * depth observation. The task is to filter out incorrect surfels, and reset
	 * both live/reference geometry to filtered ones. The fusion part should be 
	 * the same as geometry updater, but dont check the attachment and compression of
	 * appended surfels (accept all).
	 */
	class GeometryReinitProcessor {
	private:
		//The read-write access to surfel geometry
		SurfelGeometry::Ptr m_surfel_geometry[2];

		//For any processing iteration, this variable should be constant, only assign by external variable
		int m_updated_idx; // 0 ro 1, which geometry is updated read from
		float m_current_time;

		//The map from the renderer
		Renderer::FusionMaps m_fusion_maps;

		//The observation from depth camera
		CameraObservation m_observation;
		mat34 m_world2camera;


	public:
		using Ptr = std::shared_ptr<GeometryReinitProcessor>;
		GeometryReinitProcessor(SurfelGeometry::Ptr surfel_geometry[2]); //Init with double buffer
		~GeometryReinitProcessor();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(GeometryReinitProcessor);

		//The process input
		void SetInputs(
			const Renderer::FusionMaps& maps,
			const CameraObservation& observation,
			int updated_idx, float current_time,
			const mat34& world2camera
		);

		//The processing inteface
		void ProcessReinitObservedOnlySerial(unsigned& num_remaining_surfel, unsigned& num_appended_surfel, cudaStream_t stream = 0);
		void ProcessReinitNodeErrorSerial(
			unsigned& num_remaining_surfel, unsigned& num_appended_surfel, 
			const NodeAlignmentError& node_error, 
			float threshold = 0.06f, 
			cudaStream_t stream = 0
		);

		//Also, attempt to fuse the observation into surfel array at first
	private:
		SurfelFusionHandler::Ptr m_surfel_fusion_handler;
	public:
		void FuseCameraObservationNoSync(cudaStream_t stream = 0);
		DeviceArrayView<unsigned> GetRemainingSurfelIndicator() const { return m_surfel_fusion_handler->GetRemainingSurfelIndicator().ArrayView(); }
		DeviceArrayView<unsigned> GetAppendedSurfelIndicator() const { return m_surfel_fusion_handler->GetAppendedObservationCandidateIndicator(); }
		
		
		/* The remaining marker
		 */
	private:
		ReinitRemainingSurfelMarker::Ptr  m_remaining_surfel_marker;
	public:
		void MarkRemainingSurfelObservedOnly(cudaStream_t stream = 0);
		void MarkRemainingSurfelNodeError(
			const NodeAlignmentError& node_error, 
			float threshold = 0.06f,
			cudaStream_t stream = 0
		);
		
		
		/* The prefixsum for both the remaining surfel and appended surfel
		 */
	private:
		PrefixSum m_remaining_indicator_prefixsum;
		PrefixSum m_appended_indicator_prefixsum;
		void processRemainingIndicatorPrefixsum(cudaStream_t stream = 0);
		void processAppendedIndicatorPrefixsum(cudaStream_t stream = 0);
		RemainingLiveSurfel getCompactionRemainingSurfel() const;
		ReinitAppendedObservationSurfel getCompactionAppendedSurfel() const;
		
		/* The compactor buffer
		 */
	private:
		DoubleBufferCompactor::Ptr m_surfel_compactor;
	public:
		void CompactSurfelToAnotherBufferSync(unsigned& num_remaining_surfels, unsigned& num_appended_surfels, cudaStream_t stream = 0);
	};

}