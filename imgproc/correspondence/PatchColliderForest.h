#pragma once
#include "common/common_types.h"
#include "common/custom_type_traits.h"
#include "common/Serializer.h"
#include "common/Stream.h"
#include "imgproc/correspondence/gpc_common.h"

#include <memory>

namespace surfelwarp {
	
	template<int FeatureDim = 18, int NumTrees = 5>
	class PatchColliderForest {
	private:
		//The host memory for gpc nodes
		std::vector<GPCNode<FeatureDim>> m_tree_nodes_h[NumTrees];
		
		//the device memory for gpc nodes
		DeviceArray<GPCNode<FeatureDim>> m_tree_nodes_d[NumTrees];

		//These are for all the trees
		unsigned m_max_num_nodes;
		unsigned m_max_level;
	
		//the interface to access the member
	public:
		inline std::vector<GPCNode<FeatureDim>>& NodesForTree(int i) {
			return m_tree_nodes_h[i];
		}
	
	public:
		//Accessed by pointer on host
		using Ptr = std::shared_ptr<PatchColliderForest>;

		//Zero init and destroy
		PatchColliderForest();
		~PatchColliderForest() = default;

		//Access on device
		struct GPCForestDevice {
			GPCTree<FeatureDim> trees[NumTrees];
		};

		//Construct the struct
		GPCForestDevice OnDevice() const;

		//Update the maximum level
		void UpdateSearchLevel(int max_level);

		/* The method to save and load from stream
		 */
		void Save(Stream* stream) const;
		bool Load(Stream* stream);
		void UploadToDevice();
	};

	//This class implements save and load
	template<int FeatureDim, int NumTrees>
	struct has_inclass_saveload<PatchColliderForest<FeatureDim, NumTrees>> {
		static const bool value = true;
	};
}

// The implementation field
#include "imgproc/correspondence/PatchColliderForest.hpp"