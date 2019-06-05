//
// Created by wei on 3/10/18.
//

#pragma once

#include "common/common_types.h"
#include "common/algorithm_types.h"
#include "imgproc/correspondence/ImagePairCorrespondence.h"
#include "common/Constants.h"
#include "common/Stream.h"
#include "common/Serializer.h"
#include "common/BinaryFileStream.h"
#include "imgproc/correspondence/PatchColliderForest.h"
#include "common/DeviceBufferArray.h"

namespace surfelwarp {
	
	//The inference class for the Global Patch Collider
	//algorithm, this version only use rgb image pair
	class PatchColliderRGBCorrespondence : public ImagePairCorrespondence {
	public:
		//integer paramters
		enum Parameters {
			patch_radius = 10,
			patch_clip = patch_radius,
			feature_dim = 18,
			num_trees = 5,
			max_search_level = 16,
			patch_stride = 2,
			max_num_correspondence = 40000
		};
		
	private:
		//Only accept rbg images
		unsigned m_rgb_rows, m_rgb_cols;
		cudaTextureObject_t rgb_0_, rgb_1_;
		cudaTextureObject_t m_foreground_1;

		//On the original rgb image, build key-value
		//pairs with stride patch_tride, from/to where
		//the whole patch is contained in the image
		unsigned m_kvmap_rows, m_kvmap_cols;
		
		//The forest
		PatchColliderForest<feature_dim, num_trees> m_forest;
		
		//Require to sort the element
		//The key is the hashed leaf index, and the
		//value is the encoded pixel pair
		KeyValueSort<unsigned, unsigned> m_collide_sort;
		DeviceArray<unsigned> m_treeleaf_key;
		DeviceArray<unsigned> m_pixel_value;

		//The buffer for valid treeleaf key indicator
		DeviceArray<unsigned> m_candidate_pixelpair_indicator;
		
		//The buffer for compact the candidate buffer
		PrefixSum m_prefixsum;
		unsigned* m_candidate_size_pagelock;
		
		//The buffer and array for corresponded pixels
		DeviceBufferArray<ushort4> m_correspondence_pixels;
	public:
		using Ptr = std::shared_ptr<PatchColliderRGBCorrespondence>;
		
		//Do not do anything in constructor
		PatchColliderRGBCorrespondence();
		~PatchColliderRGBCorrespondence() = default;

		//The allocator and deallocator
		void AllocateBuffer(unsigned img_rows, unsigned img_cols) override;
		void ReleaseBuffer() override;

		//The main interface
		void SetInputImages(cudaTextureObject_t rgb_0, cudaTextureObject_t rgb_1, cudaTextureObject_t foreground_1) override;
		void SetInputImages(
			cudaTextureObject_t rbg_0, cudaTextureObject_t rgb_1, 
			cudaTextureObject_t depth_0, cudaTextureObject_t depth_1
		) override;
		void FindCorrespondence(cudaStream_t stream) override;

		//The return array should be zero initialized
		DeviceArray<ushort4> CorrespondedPixelPairs() const override {
			return m_correspondence_pixels.Array();
		}
	};
}
