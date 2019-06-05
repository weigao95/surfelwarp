//
// Created by wei on 3/24/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/SynchronizeArray.h"
#include <memory>

namespace surfelwarp {
	
	class KNNSearch {
	public:
		using Ptr = std::shared_ptr<KNNSearch>;
		KNNSearch() = default;
		virtual ~KNNSearch() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(KNNSearch);
		
		//Explicit allocate
		virtual void AllocateBuffer(unsigned max_num_points) = 0;
		virtual void ReleaseBuffer() = 0;
		
		//Explicit build search index
		virtual void BuildIndex(const DeviceArrayView<float4>& nodes, cudaStream_t stream = 0) = 0;
		virtual void BuildIndexHostNodes(const std::vector<float4>& nodes, cudaStream_t stream = 0) {
			LOG(FATAL) << "The index doesnt use host array, should use device array instread";
		}
		

		//Perform searching
		virtual void Skinning(
			const DeviceArrayView<float4>& vertex,
			DeviceArraySlice<ushort4> knn,
			DeviceArraySlice<float4> knn_weight,
			cudaStream_t stream = 0
		) = 0;
		virtual void Skinning(
			const DeviceArrayView<float4>& vertex, const DeviceArrayView<float4>& node,
			DeviceArraySlice<ushort4> vertex_knn, DeviceArraySlice<ushort4> node_knn,
			DeviceArraySlice<float4> vertex_knn_weight, DeviceArraySlice<float4> node_knn_weight,
			cudaStream_t stream = 0
		) = 0;

		//The checking function for KNN search
		static void CheckKNNSearch(
			const DeviceArrayView<float4>& nodes, 
			const DeviceArrayView<float4>& vertex,
			const DeviceArrayView<ushort4>& knn
		);
		
		//The result may not exactly correct, but
		//the distance should be almost the same
		static void CheckApproximateKNNSearch(
			const DeviceArrayView<float4>& nodes,
			const DeviceArrayView<float4>& vertex,
			const DeviceArrayView<ushort4>& knn
		);
	};
	
}
