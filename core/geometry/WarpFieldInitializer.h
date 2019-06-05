//
// Created by wei on 5/10/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/ArrayView.h"
#include "common/SynchronizeArray.h"
#include "core/geometry/VoxelSubsampler.h"
#include "core/WarpField.h"
#include <memory>

namespace surfelwarp {
	
	
	/**
	 * \brief The warp field initializer class takes input from reference vertex array,
	 * performing subsampling, and build the synched array of reference nodes and node SE3.
	 * The reference nodes requires more subsampling, and the SE3 is just identity.
	 * This operation is used by both the first frame and geometry reinitialization.
	 */
	class WarpFieldInitializer {
	public:
		using Ptr = std::shared_ptr<WarpFieldInitializer>;
		WarpFieldInitializer();
		~WarpFieldInitializer();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(WarpFieldInitializer);

		//The processing interface
		void InitializeReferenceNodeAndSE3FromVertex(const DeviceArrayView<float4>& reference_vertex, WarpField::Ptr warp_field, cudaStream_t stream = 0);


		/* Perform subsampling of the reference vertex, fill the node to node_candidate
		 * Of course, this operation requires sync
		 */
	private:
		VoxelSubsampler::Ptr m_vertex_subsampler;
		SynchronizeArray<float4> m_node_candidate;
		void performVertexSubsamplingSync(const DeviceArrayView<float4>& reference_vertex, cudaStream_t stream = 0);
	};
	
}
