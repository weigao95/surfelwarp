//
// Created by wei on 5/9/18.
//

#pragma once

#include "core/WarpField.h"
#include <Eigen/Core>

namespace surfelwarp {
	
	//The method updater has full access to warp field
	class WarpFieldUpdater {
	private:
		/* Compute the SE3 at point using host node se3 and reference coordinate.
		 * Assume the node se3 has been synced to host
		 */
		static bool ComputeSE3AtPointHost(
			const WarpField& warp_field,
			const float4& point,
			DualQuaternion& dq,
			ushort4& knn,
			float4& knn_weight
		);
		
		
	public:
		
		/* This method initialize the reference node from node candidate,
		 * node se3 to indentity, and upload them to device
		 */
		static void InitializeReferenceNodesAndSE3FromCandidates(
			WarpField& warp_field,
			const std::vector<float4>& node_candidate,
			cudaStream_t stream = 0
		);
		
		
		/* This method take input from a set of candidate that is not covered
		 * by current nodes, assume the Node SE3 is ready on host.
		 */
		static void UpdateWarpFieldFromUncoveredCandidate(
			WarpField& warp_field,
			const std::vector<float4>& node_candidate,
			cudaStream_t stream = 0
		);
	
	private:
		static void CheckUpdatedWarpField(
			const WarpField& warp_field,
			const std::vector<ushort4>& h_added_knn,
			const std::vector<float4>& h_added_knnweight
		);
	};
	
}
