//
// Created by wei on 5/11/18.
//

#pragma once

#include "common/algorithm_types.h"
#include "core/WarpField.h"

#include <memory>

namespace surfelwarp {
	

	/**
	 * \brief The extender class takes input from current vertex, compute the 
	 *        coverage using the reference nodes in warp field, select uncovered
	 *        vertex and hand in them to warp field for the update of reference nodes
	 *        and node SE3. The knn/weight, live nodes are resized but no updated.
	 *        The method assume the node SE3 for existing nodes are ready.
	 */
	class WarpFieldExtender {
	public:
		using Ptr = std::shared_ptr<WarpFieldExtender>;
		WarpFieldExtender();
		~WarpFieldExtender();
		SURFELWARP_NO_COPY_ASSIGN(WarpFieldExtender);

		//The processing interface
		void ExtendReferenceNodesAndSE3Sync(
			const DeviceArrayView<float4>& reference_vertex,
			const DeviceArrayView<ushort4>& vertex_knn,
			WarpField::Ptr& warp_field,
			cudaStream_t stream = 0
		);
		
		
		/* Choose node candidate from the input reference vertex
		 * do compaction and collect them into array. This require sync
		 */
	private:
		DeviceBufferArray<unsigned> m_candidate_validity_indicator;
		PrefixSum m_validity_indicator_prefixsum;
		SynchronizeArray<float4> m_candidate_vertex_array;
	
	public:
		//Label the potential candidate using vertex and knn, this method will not sync
		void labelCollectUncoveredNodeCandidate(
			const DeviceArrayView<float4>& vertex_array,
			const DeviceArrayView<ushort4>& vertex_knn,
			const DeviceArrayView<float4>& node_coordinates,
			cudaStream_t stream = 0
		);
		
		//Collect the labeled candidate, sync and query the size
		void syncQueryUncoveredNodeCandidateSize(cudaStream_t stream = 0);
	};

}
