#pragma once
#include "common/logging.h"
//#include "imgproc/correspondence/PatchColliderForest.h"

template<int FeatureDim, int NumTrees>
surfelwarp::PatchColliderForest<FeatureDim, NumTrees>::PatchColliderForest() 
: m_max_num_nodes(0), m_max_level(0)
{
	//Zero-initialize all the entities
	for(auto i = 0; i < NumTrees; i++) {
		m_tree_nodes_h[i].clear();
		m_tree_nodes_d[i] = DeviceArray<GPCNode<FeatureDim>>(nullptr, 0);
	}
}

template<int FeatureDim, int NumTrees>
typename surfelwarp::PatchColliderForest<FeatureDim, NumTrees>::GPCForestDevice surfelwarp::PatchColliderForest<FeatureDim, NumTrees>::OnDevice() const
{
	//Check if the memory has been upload to gpu memory
	if(m_tree_nodes_d[0].size() == 0) {
		LOG(FATAL) << "The forest has not been uploaded to device";
	}

	//Now we can safely assign the pointer
	GPCForestDevice forest_device;

	//Construct the tree
	GPCTree<FeatureDim> tree;
	tree.max_level = m_max_level;
	for(auto i = 0; i < NumTrees; i++) {
		tree.num_nodes = m_tree_nodes_d[i].size();
		tree.nodes = (GPCNode<FeatureDim>*) m_tree_nodes_d[i].ptr();
		forest_device.trees[i] = tree;
	}

	return forest_device;
}


template<int FeatureDim, int NumTrees>
inline void surfelwarp::PatchColliderForest<FeatureDim, NumTrees>::UpdateSearchLevel(int max_level){
	m_max_level = max_level;
}

template <int FeatureDim, int NumTrees>
void surfelwarp::PatchColliderForest<FeatureDim, NumTrees>::Save(Stream* stream) const
{
	stream->SerializeWrite(m_max_num_nodes);
	stream->SerializeWrite(m_max_level);
	for(auto i = 0; i < NumTrees; i++) {
		stream->SerializeWrite(m_tree_nodes_h[i]);
	}
}

template <int FeatureDim, int NumTrees>
bool surfelwarp::PatchColliderForest<FeatureDim, NumTrees>::Load(Stream* stream)
{
	stream->SerializeRead(&m_max_num_nodes);
	stream->SerializeRead(&m_max_level);
	m_max_num_nodes = 0;
	for(auto i = 0; i < NumTrees; i++) {
		stream->SerializeRead(&(m_tree_nodes_h[i]));
		if(m_tree_nodes_h[i].size() > m_max_num_nodes) {
			m_max_num_nodes = m_tree_nodes_h[i].size();
		}
	}
	return true;
}

template<int FeatureDim, int NumTrees>
void surfelwarp::PatchColliderForest<FeatureDim, NumTrees>::UploadToDevice() {
	for(auto i = 0; i < NumTrees; i++) {
		m_tree_nodes_d[i].upload(m_tree_nodes_h[i]);
	}
}
