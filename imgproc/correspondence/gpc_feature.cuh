#pragma once
#include "common/color_transfer.h"
#include "imgproc/correspondence/gpc_feature.h"

template<int PatchHalfSize=10>
__device__ __forceinline__
void surfelwarp::buildDCTPatchFeature(
	cudaTextureObject_t normalized_rgb, 
	int center_x, int center_y, 
	GPCPatchFeature<18>& feature
) {
	const float pi = 3.1415926f;
	const int left_x = center_x - PatchHalfSize;
	const int top_y = center_y - PatchHalfSize;

	//Load the patch into local memory
	float3 patch_ycrcb[2 * PatchHalfSize][2 * PatchHalfSize];
	for(int y = top_y; y < top_y + 2 * PatchHalfSize; y++) {
		for(int x = left_x; x < left_x + 2 * PatchHalfSize; x++)
		{
			//Read the texture
			const float4 rgba = tex2D<float4>(normalized_rgb, x, y);

			//Transfer the format
			float3 ycrcb;
			normalized_rgba2ycrcb(rgba, ycrcb);
			ycrcb.x *= 255;
			ycrcb.y *= 255;
			ycrcb.z *= 255;

			//Note the y index is at first
			patch_ycrcb[y - top_y][x - left_x] = ycrcb;
		}
	}

	//The dct iteration loop
	for(auto n0 = 0; n0 < 4; n0++) {
		for(auto n1 = 0; n1 < 4; n1++) {
			float dct_sum = 0.0f;
			for(auto y = 0; y < 2 * PatchHalfSize; y++) {
				for(auto x = 0; x < 2 * PatchHalfSize; x++)
				{
					//Read the texture
					const float3 ycrcb = patch_ycrcb[y][x];
					dct_sum += ycrcb.x
					           * cosf(pi * (x + 0.5f) * n0 / (2 * PatchHalfSize))
					           * cosf(pi * (y + 0.5f) * n1 / (2 * PatchHalfSize));
				}
			}
			//Save to patch
			feature.feature[n0 * 4 + n1] = dct_sum / float(PatchHalfSize);
		}
	}

	//Scale the descriptor
	for(auto k = 0; k < 4; k++) {
		feature.feature[k] *= 0.7071067811865475;
		feature.feature[k * 4] *= 0.7071067811865475;
	}


	//The last 2 descriptor
	float cr_sum = 0.0f;
	float cb_sum = 0.0f;
	for(auto y = 0; y < 2 * PatchHalfSize; y++) {
		for(auto x = 0; x < 2 * PatchHalfSize; x++) {
			//Read the texture
			const float3 ycrcb = patch_ycrcb[y][x];
			cr_sum += ycrcb.y;
			cb_sum += ycrcb.z;
		}
	}
	feature.feature[16] = cr_sum / (2 * PatchHalfSize);
	feature.feature[17] = cb_sum / (2 * PatchHalfSize);
}