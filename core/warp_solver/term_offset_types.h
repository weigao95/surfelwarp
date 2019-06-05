//
// Created by wei on 4/3/18.
//

#pragma once

#include "common/ArrayView.h"
#include <vector_types.h>

namespace surfelwarp {
	
	
	enum class TermType {
		DenseImage = 0,
		Smooth = 1,
		Foreground = 2,
		Feature = 3,
		//DensityMap = 4,
		Invalid = 5
	};
	
	struct TermTypeOffset {
		unsigned offset_value[4];
		
		//The accessed interface
		__host__ __device__ __forceinline__ const unsigned& operator[](const int idx) const {
			return offset_value[idx];
		}
		
		//The size of terms
		__host__ __device__ __forceinline__ unsigned TermSize() const { return offset_value[3]; }
		__host__ __forceinline__ unsigned ScalarTermSize() const {
			return DenseImageTermSize() + ForegroundTermSize() + 3 * (SmoothTermSize() + FeatureTermSize());
		}
		
		//The different type of terms
		__host__ __device__ __forceinline__ unsigned DenseImageTermSize() const { return offset_value[0]; }
		__host__ __device__ __forceinline__ unsigned SmoothTermSize() const { return offset_value[1] - offset_value[0]; }
		__host__ __device__ __forceinline__ unsigned ForegroundTermSize() const { return offset_value[2] - offset_value[1]; }
		__host__ __device__ __forceinline__ unsigned FeatureTermSize() const { return offset_value[3] - offset_value[2]; }
	};
	
	
	inline void size2offset(
		TermTypeOffset& offset,
		DeviceArrayView<ushort4> dense_depth_knn,
		DeviceArrayView<ushort2> node_graph,
		//These costs might be empty
		DeviceArrayView<ushort4> foreground_mask_knn = DeviceArrayView<ushort4>(),
		DeviceArrayView<ushort4> sparse_feature_knn  = DeviceArrayView<ushort4>()
	) {
		unsigned prefix_sum = 0;
		prefix_sum += dense_depth_knn.Size();
		offset.offset_value[0] = prefix_sum;
		prefix_sum += node_graph.Size();
		offset.offset_value[1] = prefix_sum;
		prefix_sum += foreground_mask_knn.Size();
		offset.offset_value[2] = prefix_sum;
		prefix_sum += sparse_feature_knn.Size();
		offset.offset_value[3] = prefix_sum;
	}
	
	__host__ __device__ __forceinline__
	void query_typed_index(unsigned term_idx, const TermTypeOffset& offset, TermType& type, unsigned& typed_idx)
	{
		if(term_idx < offset[0]) {
			type = TermType::DenseImage;
			typed_idx = term_idx - 0;
			return;
		}
		if(term_idx >= offset[0] && term_idx < offset[1]) {
			type = TermType::Smooth;
			typed_idx = term_idx - offset[0];
			return;
		}
		if(term_idx >= offset[1] && term_idx < offset[2]) {
			type = TermType::Foreground;
			typed_idx = term_idx - offset[1];
			return;
		}
		if(term_idx >= offset[2] && term_idx < offset[3]) {
			type = TermType::Feature;
			typed_idx = term_idx - offset[2];
			return;
		}

		//Not a valid term
		type = TermType::Invalid;
		typed_idx = 0xFFFFFFFF;
	}

	__host__ __device__ __forceinline__
	void query_typed_index(unsigned term_idx, const TermTypeOffset& offset, TermType& type, unsigned& typed_idx, unsigned& scalar_term_idx)
	{
		unsigned scalar_offset = 0;
		if(term_idx < offset[0]) {
			type = TermType::DenseImage;
			typed_idx = term_idx - 0;
			scalar_term_idx = term_idx;
			return;
		}

		scalar_offset += offset.DenseImageTermSize();
		if(term_idx >= offset[0] && term_idx < offset[1]) {
			type = TermType::Smooth;
			typed_idx = term_idx - offset[0];
			scalar_term_idx = scalar_offset + 3 * typed_idx;
			return;
		}

		scalar_offset += 3 * offset.SmoothTermSize();
		if(term_idx >= offset[1] && term_idx < offset[2]) {
			type = TermType::Foreground;
			typed_idx = term_idx - offset[1];
			scalar_term_idx = typed_idx + scalar_offset;
			return;
		}

		scalar_offset += offset.ForegroundTermSize();
		if(term_idx >= offset[2] && term_idx < offset[3]) {
			type = TermType::Feature;
			typed_idx = term_idx - offset[2];
			scalar_term_idx = scalar_offset + 3 * typed_idx;
			return;
		}

		//Not a valid term
		type = TermType::Invalid;
		typed_idx = 0xFFFFFFFF;
		scalar_term_idx = 0xFFFFFFFF;
	}


	__host__ __device__ __forceinline__
	void query_nodepair_index(unsigned term_idx, const TermTypeOffset& offset, TermType& type, unsigned& typed_idx, unsigned& nodepair_idx)
	{
		unsigned pair_offset = 0;
		if(term_idx < offset[0]) {
			type = TermType::DenseImage;
			typed_idx = term_idx - 0;
			nodepair_idx = pair_offset + 6 * term_idx;
			return;
		}

		pair_offset += 6 * offset.DenseImageTermSize();
		if(term_idx >= offset[0] && term_idx < offset[1]) {
			type = TermType::Smooth;
			typed_idx = term_idx - offset[0];
			nodepair_idx = pair_offset + 1 * typed_idx;
			return;
		}

		pair_offset += 1 * offset.SmoothTermSize();
		if(term_idx >= offset[1] && term_idx < offset[2]) {
			type = TermType::Foreground;
			typed_idx = term_idx - offset[1];
			nodepair_idx = pair_offset + 6 * typed_idx;
			return;
		}

		pair_offset += 6 * offset.ForegroundTermSize();
		if(term_idx >= offset[2] && term_idx < offset[3]) {
			type = TermType::Feature;
			typed_idx = term_idx - offset[2];
			nodepair_idx = pair_offset + 6 * typed_idx;
			return;
		}

		//Not a valid term
		type = TermType::Invalid;
		typed_idx = 0xFFFFFFFF;
		nodepair_idx = 0xFFFFFFFF;
	}
}
