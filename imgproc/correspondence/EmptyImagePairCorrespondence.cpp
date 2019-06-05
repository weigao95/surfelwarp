//
// Created by wei on 3/7/18.
//

#include "imgproc/correspondence/EmptyImagePairCorrespondence.h"

void surfelwarp::EmptyImagePairCorrespondence::AllocateBuffer(unsigned img_rows, unsigned img_cols)
{
}

void surfelwarp::EmptyImagePairCorrespondence::ReleaseBuffer()
{
}

void surfelwarp::EmptyImagePairCorrespondence::SetInputImages(
	cudaTextureObject_t rgb_0, 
	cudaTextureObject_t rgb_1, 
	cudaTextureObject_t foreground_1
) {
}

void surfelwarp::EmptyImagePairCorrespondence::FindCorrespondence(cudaStream_t stream)
{
}

surfelwarp::DeviceArray<ushort4> surfelwarp::EmptyImagePairCorrespondence::CorrespondedPixelPairs() const
{
	return DeviceArray<ushort4>();
}

void surfelwarp::EmptyImagePairCorrespondence::SetInputImages(
	cudaTextureObject_t rbg_0, cudaTextureObject_t rgb_1,
	cudaTextureObject_t depth_0, cudaTextureObject_t depth_1
) {

}
