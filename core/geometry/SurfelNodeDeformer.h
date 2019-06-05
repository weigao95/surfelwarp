//
// Created by wei on 5/7/18.
//

#pragma once

#include "common/macro_utils.h"
#include "core/SurfelGeometry.h"
#include "core/WarpField.h"
#include <memory>


namespace surfelwarp {
	
	
	/* The deformer will perform forward and inverse
	 * warping given the geometry and warp field. It has
	 * full accessed to the SurfelGeometry instance.
	 */
	class SurfelNodeDeformer {
	public:
		//The processing of forward warp, may
		//use a node se3 different from the on in warp field
		static void ForwardWarpSurfelsAndNodes(
			WarpField& warp_field,
			SurfelGeometry& geometry,
			const DeviceArrayView<DualQuaternion>& node_se3,
			cudaStream_t stream = 0
		);
		static void ForwardWarpSurfelsAndNodes(
			WarpField& warp_field,
			SurfelGeometry& geometry,
			cudaStream_t stream = 0
		);


		//The processing interface, may use a node se3
		//different from the one in warp field
		static void InverseWarpSurfels(
			SurfelGeometry& geometry,
			const DeviceArrayView<DualQuaternion>& node_se3,
			cudaStream_t stream = 0
		);
		static void InverseWarpSurfels(
			const WarpField& warp_field,
			SurfelGeometry& geometry,
			const DeviceArrayView<DualQuaternion>& node_se3,
			cudaStream_t stream = 0
		);
		static void InverseWarpSurfels(
			const WarpField& warp_field,
			SurfelGeometry& geometry,
			cudaStream_t stream = 0
		);
		
		//Check the size of the geometry
		static void CheckSurfelGeometySize(const SurfelGeometry& geometry);
	};
	
}
