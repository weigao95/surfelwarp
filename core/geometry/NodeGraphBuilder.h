//
// Created by wei on 3/24/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/ArrayView.h"
#include "common/DeviceBufferArray.h"
#include <memory>

namespace surfelwarp {
	
	class NodeGraphBuilder {
	public:
		using Ptr = std::shared_ptr<NodeGraphBuilder>;
		NodeGraphBuilder() = default;
		virtual ~NodeGraphBuilder() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(NodeGraphBuilder);
		
		//Might need to assign or release
		virtual void BuildNodeGraph(
			const DeviceArrayView<float4>& reference_nodes,
			DeviceBufferArray<ushort2>& node_graph,
			cudaStream_t stream = 0
		) = 0;
		
		//By default, just rebuild the graph
		virtual void UpdateNodeGraph(
			const DeviceArrayView<float4>& reference_nodes,
			size_t newnode_offset,
			DeviceBufferArray<ushort2>& node_graph,
			cudaStream_t stream = 0)
		{ BuildNodeGraph(reference_nodes, node_graph, stream); }
	};
	
}
