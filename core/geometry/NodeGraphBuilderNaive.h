//
// Created by wei on 3/24/18.
//

#pragma once

#include "core/geometry/NodeGraphBuilder.h"

namespace surfelwarp {
	
	class NodeGraphBuilderNaive : public NodeGraphBuilder {
	public:
		using Ptr = std::shared_ptr<NodeGraphBuilderNaive>;
		SURFELWARP_DEFAULT_CONSTRUCT_DESTRUCT(NodeGraphBuilderNaive);
		SURFELWARP_NO_COPY_ASSIGN_MOVE(NodeGraphBuilderNaive);

		void BuildNodeGraph(
			const DeviceArrayView<float4>& reference_nodes, 
			DeviceBufferArray<ushort2>& node_graph, 
			cudaStream_t stream
		) override;

		//In fact, this version is stateless
		static void buildNodeGraph(
			const DeviceArrayView<float4>& reference_nodes, 
			DeviceArraySlice<ushort2> node_graph, 
			cudaStream_t stream
		);
	};
}
