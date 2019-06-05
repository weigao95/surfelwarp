#pragma once
#include "common/macro_utils.h"
#include "common/ArrayView.h"
#include "core/geometry/KNNSearch.h"

namespace surfelwarp {
	
	class KNNBruteForceReferenceNodes : public KNNSearch {
	private:
		//Access only the global singleton
		KNNBruteForceReferenceNodes();
	public:
		using Ptr = std::shared_ptr<KNNBruteForceReferenceNodes>;
		static KNNBruteForceReferenceNodes::Ptr Instance();
		~KNNBruteForceReferenceNodes();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(KNNBruteForceReferenceNodes);

		//Do not need allocate/deallocate
		void AllocateBuffer(unsigned max_num_points) override;
		void ReleaseBuffer() override;

		//Record the nodes into points_view
		void BuildIndex(const DeviceArrayView<float4>& nodes, cudaStream_t stream) override;
		void UpdateIndex(
			const SynchronizeArray<float4>& nodes,
			size_t new_nodes_offset,
			cudaStream_t stream
		);
	
	private:
		//Assigned in build/update index
		unsigned short m_num_nodes;
		
		//The buffer to clear the node coordinates
		DeviceArray<float4> m_invalid_nodes;
		void clearConstantPoints(cudaStream_t stream = 0);
		
		//Switch the index from one set of nodes to another
		//Assuming the size of node is the same
		void replaceWithMorePoints(
			const DeviceArrayView<float4>& nodes,
			cudaStream_t stream = 0
		);
	
	public:
		//The search interface
		void Skinning(
			const DeviceArrayView<float4>& vertex, 
			DeviceArraySlice<ushort4> knn, 
			DeviceArraySlice<float4> knn_weight,
			cudaStream_t stream = 0
		) override;

		void Skinning(
			const DeviceArrayView<float4>& vertex,
			const DeviceArrayView<float4>& node, 
			DeviceArraySlice<ushort4> vertex_knn, 
			DeviceArraySlice<ushort4> node_knn, 
			DeviceArraySlice<float4> vertex_knn_weight,
			DeviceArraySlice<float4> node_knn_weight,
			cudaStream_t stream = 0
		) override;
	};

}
