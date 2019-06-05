//
// Created by wei on 5/10/18.
//

#pragma once

#include "common/ArrayView.h"
#include "core/geometry/KNNSearch.h"
#include "core/SurfelGeometry.h"
#include "core/WarpField.h"
#include <memory>

namespace surfelwarp {
	
	/**
	 * \brief This class is a singleton. It maintains a brute force version for skinning update,
	 * and another knn searcher for (re)initialization. When there is no other skinner available,
	 * the skinning at (re)initialization will also be performed using brute for version
	 */
	class ReferenceNodeSkinner {
	private:
		//Do not allow outclass construction
		ReferenceNodeSkinner();
	public:
		using Ptr = std::shared_ptr<ReferenceNodeSkinner>;
		static std::shared_ptr<ReferenceNodeSkinner> Instance();
		~ReferenceNodeSkinner();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(ReferenceNodeSkinner);



		/* The method for initial skinning. Re-build the whole index, including
		 * the m_init_skinner and the brute force skinner.
		 */
	private:
		KNNSearch::Ptr m_init_skinner; //Use for initialization, may need to build index
		unsigned m_num_bruteforce_nodes; //The number of nodes recorded in the brute force skinning index
		DeviceArray<float4> m_invalid_nodes;
		
		//The method to (re)build the skinning index
		void fillInvalidConstantPoints(cudaStream_t stream = 0);
		void replaceWithMorePoints(
			const DeviceArrayView<float4>& nodes,
			cudaStream_t stream = 0
		);
		void buildBruteForceIndex(const DeviceArrayView<float4>& nodes, cudaStream_t stream);
		
		//The workforce function for initial skinning
		void performBruteForceSkinning(
			const DeviceArrayView<float4>& reference_vertex,
			const DeviceArrayView<float4>& reference_node,
			DeviceArraySlice<ushort4> vertex_knn,
			DeviceArraySlice<ushort4> node_knn,
			DeviceArraySlice<float4> vertex_knn_weight,
			DeviceArraySlice<float4> node_knn_weight,
			cudaStream_t stream = 0
		) const;
	public:
		//Clear and rebuild both the brute-force and init_index (if not nullptr)
		void BuildInitialSkinningIndex(const SynchronizeArray<float4>& nodes, cudaStream_t stream = 0);
		void PerformSkinning(SurfelGeometry::SkinnerInput geometry, WarpField::SkinnerInput warp_field, cudaStream_t stream = 0);


		/* The method for index and skinning update. Only perform for brute-force index. 
		 * The init_skinner does not need to be updated.
		 */
	private:
		//The workforce function
		void updateSkinning(
			unsigned newnode_offset,
			const DeviceArrayView<float4>& reference_vertex,
			const DeviceArrayView<float4>& reference_node,
			DeviceArraySlice<ushort4> vertex_knn,
			DeviceArraySlice<ushort4> node_knn,
			DeviceArraySlice<float4> vertex_knn_weight,
			DeviceArraySlice<float4> node_knn_weight,
			cudaStream_t stream = 0
		) const;
	public:
		//nodes[newnode_offset] should be the first new node
		void UpdateBruteForceSkinningIndexWithNewNodes(const DeviceArrayView<float4>& nodes, unsigned newnode_offset, cudaStream_t stream = 0);
		void PerformSkinningUpdate(
			SurfelGeometry::SkinnerInput geometry,
			WarpField::SkinnerInput warp_field,
			unsigned newnode_offset,
			cudaStream_t stream = 0
		);
	};

}