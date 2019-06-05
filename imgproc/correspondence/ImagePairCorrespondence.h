//
// Created by wei on 3/7/18.
//

#pragma once

#include "common/common_types.h"
#include <memory>

namespace surfelwarp {

	class ImagePairCorrespondence {
	public:
		//Access by pointer
		using Ptr = std::shared_ptr<ImagePairCorrespondence>;

		//Default contruction/de-contruction, no copy/assign/move
		ImagePairCorrespondence() = default;
		virtual ~ImagePairCorrespondence() = default;
		ImagePairCorrespondence(const ImagePairCorrespondence&) = delete;
		ImagePairCorrespondence(ImagePairCorrespondence&&) = delete;
		ImagePairCorrespondence& operator=(const ImagePairCorrespondence&) = delete;
		ImagePairCorrespondence& operator=(ImagePairCorrespondence&&) = delete;
		
		//Stand-alone buffer allocator/release, the caller's
		//responsibility to ensure one-allocate-one-release.
		//The input is the original size of the image, while
		//the algorithm might use subsampling
		virtual void AllocateBuffer(unsigned img_rows, unsigned img_cols) = 0;
		virtual void ReleaseBuffer() = 0;

		//The operation interface
		virtual void SetInputImages(cudaTextureObject_t rgb_0, cudaTextureObject_t rgb_1, cudaTextureObject_t foreground_1) = 0;
		virtual void SetInputImages(
			cudaTextureObject_t rbg_0, cudaTextureObject_t rgb_1,
			cudaTextureObject_t depth_0, cudaTextureObject_t depth_1) = 0;
		virtual void FindCorrespondence(cudaStream_t stream = 0) = 0;

		//The accessing method.
		//For each return, x and y is the pixel coordinate in rgb_0
		//While z and w is the pixel coordinate in rgb_1. These two
		//pixels should be correspondenced. The method should return
		//a read only device array (caller won't explicit delete it).
		virtual DeviceArray<ushort4> CorrespondedPixelPairs() const = 0;
	};

}