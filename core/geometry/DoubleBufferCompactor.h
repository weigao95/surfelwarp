//
// Created by wei on 5/7/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/algorithm_types.h"
#include "common/DeviceBufferArray.h"
#include "core/geometry/fusion_types.h"
#include "core/SurfelGeometry.h"

#include <memory>


namespace surfelwarp {
	


	/**
	 * \brief The compactor takes input from:
	 * 1. Original live surfel, their knn/weights, and validity indicator
	 * 2. Appended depth surfel, their knn/weights, and validity indicator
	 * The task of the compactor is to compact all these surfels to another
	 * buffer provided by OpenGL pipeline, and count the total number of compacted surfels. 
	 */
	class DoubleBufferCompactor {
	private:
		//The appended observation surfel from the depth/color image
		AppendedObservationSurfelKNN m_appended_surfel_knn;
		
		//The append surfel for reinit
		ReinitAppendedObservationSurfel m_reinit_append_surfel;

		//The surfel from the original model
		RemainingLiveSurfel m_remaining_surfel;
		RemainingSurfelKNN m_remaining_knn;

		//The geometry that shoule be compacted to
		SurfelGeometry::Ptr m_compact_to_geometry;

		//The rows and cols used to decode the geometry
		unsigned m_image_rows, m_image_cols;
	public:
		using Ptr = std::shared_ptr<DoubleBufferCompactor>;
		DoubleBufferCompactor();
		~DoubleBufferCompactor();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(DoubleBufferCompactor);


		//The input from both append handler and geometry updater
		void SetFusionInputs(
			const RemainingLiveSurfelKNN& remaining_surfels,
			const AppendedObservationSurfelKNN& appended_surfels,
			SurfelGeometry::Ptr compacted_geometry
		);
		
		//The input from geometry reiniter
		void SetReinitInputs(
			const RemainingLiveSurfel& remaining_surfels,
			const ReinitAppendedObservationSurfel& append_surfels,
			SurfelGeometry::Ptr compact_to_geometry
		);
	
		//The main data for compaction, note that this method will sync
		//to query the size of remaining and appended sufel
		void PerformCompactionGeometryKNNSync(unsigned& num_valid_remaining_surfels, unsigned& num_valid_append_surfels, cudaStream_t stream = 0);
		void PerformComapctionGeometryOnlySync(unsigned& num_valid_remaining_surfels, unsigned& num_valid_append_surfels, const mat34& camera2world, cudaStream_t stream = 0);
	};


} 