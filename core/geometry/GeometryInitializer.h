//
// Created by wei on 3/19/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/CameraObservation.h"
#include "core/WarpField.h"
#include "core/SurfelGeometry.h"
#include "core/geometry/VoxelSubsampler.h"
#include <memory>

namespace surfelwarp {
	
	class GeometryInitializer {
	public:
		//Access by pointer
		using Ptr = std::shared_ptr<GeometryInitializer>;
		
		//Explicit allocate/deallocate
		SURFELWARP_DEFAULT_CONSTRUCT_DESTRUCT(GeometryInitializer);
		SURFELWARP_NO_COPY_ASSIGN_MOVE(GeometryInitializer);
		
		//It is caller's duty to perform allocation before
		//using the interface
		void AllocateBuffer();
		void ReleaseBuffer();
		
		//The processer interface
		void InitFromObservationSerial(
			SurfelGeometry& geometry,
			const DeviceArrayView<DepthSurfel>& surfel_array,
			cudaStream_t stream = 0
		);

		//The members from other classes
		using GeometryAttributes = SurfelGeometry::GeometryAttributes;

		/* Collect the compacted valid surfel array into geometry
		 */
	private:
		static void initSurfelGeometry(
			GeometryAttributes geometry, 
			const DeviceArrayView<DepthSurfel>& surfel_array,
			cudaStream_t stream = 0
		);
	};
}
