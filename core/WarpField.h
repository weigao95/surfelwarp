//
// Created by wei on 3/18/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/Constants.h"
#include "common/SynchronizeArray.h"
#include "common/DeviceBufferArray.h"
#include "common/sanity_check.h"
#include "math/DualQuaternion.hpp"
#include "core/geometry/KNNSearch.h"
#include "core/geometry/NodeGraphBuilder.h"

namespace surfelwarp {
	
	//Forware declaration
	class WarpFieldUpdater;
	class SurfelNodeDeformer;
	
	class WarpField {
	private:
		//The sync members that might be accessed on host
		SynchronizeArray<float4> m_reference_node_coords;
		SynchronizeArray<DualQuaternion> m_node_se3;
		
		//These property will be uploaded from host to device
		DeviceBufferArray<ushort4> m_node_knn;
		DeviceBufferArray<float4> m_node_knn_weight;
		
		//The member that will only be accessed on device
		DeviceBufferArray<float4> m_live_node_coords;
		DeviceBufferArray<ushort2> m_node_graph;

		//This class need full access to warp field
		friend class WarpFieldUpdater;
		
		//This class need to write to live node coords
		friend class SurfelNodeDeformer;

		/* The initialization of the warp field.
		 * Can not be expanded once allocated.
		 */
	private:
		void allocateBuffer(size_t max_num_nodes);
		void releaseBuffer();
	public:
		using Ptr = std::shared_ptr<WarpField>;
		WarpField();
		~WarpField();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(WarpField);
		
		//Resize all the DeviceBufferArray except node graph to the given size
		void ResizeDeviceArrayToNodeSize(unsigned node_size);
		unsigned CheckAndGetNodeSize() const;

		
		/* The method to build the node graph.
		 * Use virual class to implement various methods.
		 * Assuming the reference nodes are ready on device
		 */
	private:
		NodeGraphBuilder::Ptr m_node_graph_builder;
	public:
		void BuildNodeGraph(cudaStream_t stream = 0);
		
		
		/* Update my warp field from the solved value
		 */
	public:
		void UpdateHostDeviceNodeSE3NoSync(DeviceArrayView<DualQuaternion> node_se3, cudaStream_t stream = 0);
		void SyncNodeSE3ToHost() { m_node_se3.SynchronizeToHost(); };
		
		/* The solver accessing interface
		 */
	public:
		struct SolverInput {
			DeviceArrayView<DualQuaternion> node_se3;
			DeviceArrayView<float4> reference_node_coords;
			DeviceArrayView<ushort2> node_graph;
		};
		SolverInput SolverAccess() const;
		
		/* The input accessed by geometry updater
		 */
		struct LiveGeometryUpdaterInput {
			DeviceArrayView<float4> live_node_coords;
			DeviceArrayView<float4> reference_node_coords;
			DeviceArrayView<DualQuaternion> node_se3;
		};
		LiveGeometryUpdaterInput GeometryUpdaterAccess() const;
		
		
		/* The input accessed by skinner
		 */
		struct SkinnerInput {
			DeviceArrayView<float4> reference_node_coords;
			DeviceArraySlice<ushort4> node_knn;
			DeviceArraySlice<float4> node_knn_weight;
		};
		SkinnerInput SkinnerAccess();
		

		/* The read-only access for debugging
		 */
	public:
		const SynchronizeArray<float4>& ReferenceNodeCoordinates() const { return m_reference_node_coords; }
		DeviceArrayView<float4> LiveNodeCoordinates() const { return m_live_node_coords.ArrayView(); }
		
		
		/* The access by legacy solver
		 */
		struct LegacySolverAccess {
			DeviceArray<DualQuaternion> node_se3;
			DeviceArray<float4> reference_node_coords;
			DeviceArray<ushort2> node_graph;
		};
		LegacySolverAccess LegacySolverInput();
		
		//The the knn of the node
		void CheckNodeKNN();

		//Use the warp field inside this class, but do not touch the live nodes
		void ForwardWarpDebug(
			const DeviceArrayView<float4>& reference_vertex,
			const DeviceArrayView<ushort4>& knn,
			const DeviceArrayView<float4>& knn_weight,
			const DeviceArrayView<DualQuaternion>& node_se3,
			DeviceArraySlice<float4> live_vertex
		) const;
	};
	
	
}