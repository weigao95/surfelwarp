//
// Created by wei on 4/16/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/DeviceBufferArray.h"
#include "common/algorithm_types.h"
#include "core/warp_solver/term_offset_types.h"
#include <memory>

namespace surfelwarp {
	
	class NodePair2TermsIndex {
	public:
		using Ptr = std::shared_ptr<NodePair2TermsIndex>;
		NodePair2TermsIndex();
		~NodePair2TermsIndex() = default;
		SURFELWARP_NO_COPY_ASSIGN(NodePair2TermsIndex);
		
		//Explicit allocate/de-allocate
		void AllocateBuffer();
		void ReleaseBuffer();
		
		//The input for index
		void SetInputs(
			unsigned num_nodes,
			DeviceArrayView<ushort4> dense_image_knn,
			DeviceArrayView<ushort2> node_graph,
			//These costs might be empty
			DeviceArrayView<ushort4> foreground_mask_knn = DeviceArrayView<ushort4>(),
			DeviceArrayView<ushort4> sparse_feature_knn  = DeviceArrayView<ushort4>()
		);
		
		//The operation interface
		void BuildHalfIndex(cudaStream_t stream = 0);
		void QueryValidNodePairSize(cudaStream_t stream = 0); //Will block the stream
		unsigned NumTerms() const;
		unsigned NumKeyValuePairs() const;
		
		//Build the symmetric and row index
		void BuildSymmetricAndRowBlocksIndex(cudaStream_t stream = 0);
		
		//The access interface
		struct NodePair2TermMap {
			DeviceArrayView<unsigned> encoded_nodepair;
			DeviceArrayView<uint2> nodepair_term_range;
			DeviceArrayView<unsigned> nodepair_term_index;
			TermTypeOffset term_offset;
			//For bin-block csr
			DeviceArrayView<unsigned> blkrow_offset;
			DeviceArrayView<int> binblock_csr_rowptr;
			const int* binblock_csr_colptr;
		};
		NodePair2TermMap GetNodePair2TermMap() const;
		
		
		/* Fill the key and value given the terms
		 */
	private:
		//The input map from terms to nodes, the input might be empty for dense_density, foreground mask and sparse feature
		struct {
			DeviceArrayView<ushort4> dense_image_knn; //Each depth scalar term has 4 nearest neighbour
			DeviceArrayView<ushort2> node_graph;
			//DeviceArrayView<ushort4> density_map_knn; //Each density scalar term has 4 nearest neighbour
			DeviceArrayView<ushort4> foreground_mask_knn; //The same as density term
			DeviceArrayView<ushort4> sparse_feature_knn; //Each 4 nodes correspond to 3 scalar cost
		} m_term2node;
		
		//The term offset of term2node map
		TermTypeOffset m_term_offset;
		unsigned m_num_nodes;
		
		/* The key-value buffer for indexing
		 */
	private:
		DeviceBufferArray<unsigned> m_nodepair_keys;
		DeviceBufferArray<unsigned> m_term_idx_values;
	public:
		void buildTermKeyValue(cudaStream_t stream = 0);
		
		
		/* Perform key-value sort, do compaction
		 */
	private:
		KeyValueSort<unsigned, unsigned> m_nodepair2term_sorter;
		DeviceBufferArray<unsigned> m_segment_label;
		PrefixSum m_segment_label_prefixsum;
		
		//The compacted half key and values
		DeviceBufferArray<unsigned> m_half_nodepair_keys;
		DeviceBufferArray<unsigned> m_half_nodepair2term_offset;
	public:
		void sortCompactTermIndex(cudaStream_t stream = 0);
	
		
		/* Fill the other part of the matrix
		 */
	private:
		DeviceBufferArray<unsigned> m_compacted_nodepair_keys;
		DeviceBufferArray<uint2> m_nodepair_term_range;
		KeyValueSort<unsigned, uint2> m_symmetric_kv_sorter;
	public:
		void buildSymmetricCompactedIndex(cudaStream_t stream = 0);
		
		
		/* Compute the offset and length of each BLOCKED row
		 */
	private:
		DeviceBufferArray<unsigned> m_blkrow_offset_array;
		DeviceBufferArray<unsigned> m_blkrow_length_array;
		void blockRowOffsetSanityCheck();
		void blockRowLengthSanityCheck();
	public:
		void computeBlockRowLength(cudaStream_t stream = 0);
		
		
		/* Compute the map from block row to the elements in this row block
		 */
	private:
		DeviceBufferArray<unsigned> m_binlength_array;
		DeviceBufferArray<unsigned> m_binnonzeros_prefixsum;
		DeviceBufferArray<int> m_binblocked_csr_rowptr;
		void binLengthNonzerosSanityCheck();
		void binBlockCSRRowPtrSanityCheck();
	public:
		void computeBinLength(cudaStream_t stream = 0);
		void computeBinBlockCSRRowPtr(cudaStream_t stream = 0);
		
		
		/* Compute the column ptr for bin block csr matrix
		 */
	private:
		DeviceBufferArray<int> m_binblocked_csr_colptr;
		void binBlockCSRColumnPtrSanityCheck();
	public:
		void nullifyBinBlockCSRColumePtr(cudaStream_t stream = 0);
		void computeBinBlockCSRColumnPtr(cudaStream_t stream = 0);
		
		
		
		/* Perform sanity check for nodepair2term
		 */
	public:
		void CheckHalfIndex();
		void CompactedIndexSanityCheck();
		
		//Check the size and distribution of the size of index
		void IndexStatistics();
		
		//Check whether the smooth term contains nearly all index
		//that can be exploited to implement more efficient indexing
		//Required download data and should not be used in real-time code
		void CheckSmoothTermIndexCompleteness();
	private:
		static void check4NNTermIndex(int typed_term_idx,
		                       const std::vector<ushort4> &knn_vec,
		                       unsigned encoded_nodepair);
		static void checkSmoothTermIndex(int smooth_term_idx, const std::vector<ushort2>& node_graph, unsigned encoded_nodepair);
	};
	
}
