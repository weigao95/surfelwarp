#pragma once
#include "common/common_types.h"
#include "common/Stream.h"
#include "common/custom_type_traits.h"

namespace surfelwarp {
	
	//A small struct hold the feature of a gpc patch.
	template<int FeatureDim = 18>
	struct GPCPatchFeature {
		float feature[FeatureDim];
	};


	//A small struct for the node in the gpc tree
	template<int FeatureDim = 18> 
	struct GPCNode {
		float coefficient[FeatureDim];
		float boundary;
		int left_child, right_child;

		//The dot product with the feature
		__host__ __device__ __forceinline__
		float dot(const GPCPatchFeature<FeatureDim>& feature) const {
			float dot_value = 0.0f;
			for(auto i = 0; i < FeatureDim; i++) {
				dot_value += feature.feature[i] * coefficient[i];
			}
			return dot_value;
		}
		
		//The save and load method
		inline void Save(Stream* stream) const {
			for(auto i = 0; i < FeatureDim; i++) {
				stream->SerializeWrite<float>(coefficient[i]);
			}
			stream->SerializeWrite<float>(boundary);
			stream->SerializeWrite<int>(left_child);
			stream->SerializeWrite<int>(right_child);
		}
		
		inline bool Load(Stream* stream) {
			for(auto i = 0; i < FeatureDim; i++) {
				stream->SerializeRead<float>(&coefficient[i]);
			}
			stream->SerializeRead<float>(&boundary);
			stream->SerializeRead<int>(&left_child);
			stream->SerializeRead<int>(&right_child);
			return true;
		}
	};
	
	//Declare this class implement save and load
	template<int FeatureDim>
	struct has_inclass_saveload<GPCNode<FeatureDim>> {
		static const bool value = true;
	};

	template<int FeatureDim = 18>
	struct GPCTree {
		GPCNode<FeatureDim>* nodes;
		unsigned num_nodes;
		unsigned max_level;

		//Find the leaf for a given patch
		__host__ __device__ __forceinline__
		unsigned leafForPatch(const GPCPatchFeature<FeatureDim>& patch_feature) const {
			unsigned node_idx = 0, prev_idx = 0;

			//the main search loop is bounded by the depth
			for(auto i = 0; i < max_level; i++) {
				prev_idx = node_idx;
				//Load the node, this goona be expensive
				const GPCNode<FeatureDim>& node = nodes[node_idx];
				
				//Determine the next position
				if(node.dot(patch_feature) < node.boundary) {
					node_idx = node.right_child;
				} 
				else {
					node_idx = node.left_child;
				}

				//Check if break
				if(node_idx == 0 || node_idx >= num_nodes) break;
			}

			//prev_idx is the index with valid tree node
			return prev_idx;
		}
	};
	
	
}