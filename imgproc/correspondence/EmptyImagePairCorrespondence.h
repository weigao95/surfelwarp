#pragma once
#include "imgproc/correspondence/ImagePairCorrespondence.h"
#include <memory>

namespace surfelwarp {
	
	//Just a placeholder class, for build scripts and 
	//the parallel implementations of later codes.
	class EmptyImagePairCorrespondence : public ImagePairCorrespondence {
	public:
		//Update the pointer
		using Ptr = std::shared_ptr<EmptyImagePairCorrespondence>;

		//All default functions
		EmptyImagePairCorrespondence() = default;
		~EmptyImagePairCorrespondence() = default;
		void AllocateBuffer(unsigned img_rows, unsigned img_cols) override;
		void ReleaseBuffer() override;
		void SetInputImages(cudaTextureObject_t rgb_0, cudaTextureObject_t rgb_1, cudaTextureObject_t foreground_1) override;
		void SetInputImages(
			cudaTextureObject_t rbg_0, cudaTextureObject_t rgb_1,
			cudaTextureObject_t depth_0, cudaTextureObject_t depth_1) override;
		void FindCorrespondence(cudaStream_t stream) override;
		DeviceArray<ushort4> CorrespondedPixelPairs() const override;
	};
}