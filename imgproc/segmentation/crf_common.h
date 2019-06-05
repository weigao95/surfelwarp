//
// Created by wei on 2/24/18.
//

#pragma once

#include "common/common_types.h"
#include "math/vector_ops.hpp"

namespace surfelwarp {

	/**
	 * \brief Compute the appearance kernel between two input.
	 *        The input is assumed normalized, thus directly use
	 *        the squared norm and exp.
	 * \param feature_0 Normalized feature fector, [(x, y) / sigma_alpha, (r, g, b) / sigma_beta]
	 * \param feature_1 
	 * \return 
	 */
#if defined(__CUDA_ARCH__)
	__device__ __forceinline__
	float appearance_kernel(const float feature_0[5], const float feature_1[5])
	{
		float squared_diff = 0.0f;
		for (auto i = 0; i < 5; i++) {
			const auto diff = feature_0[i] - feature_1[i];
			squared_diff += diff * diff;
		}

		//Use fast math intrinsic
		return __expf(-0.5 * squared_diff);
	}
#else
	__host__ __forceinline__
	float appearance_kernel(const float feature_0[5], const float feature_1[5])
	{
		float squared_diff = 0.0f;
		for (auto i = 0; i < 5; i++) {
			const auto diff = feature_0[i] - feature_1[i];
			squared_diff += diff * diff;
		}
		return expf(-0.5 * squared_diff);
	}
#endif


	__host__ __device__ __forceinline__
	float smooth_kernel(const unsigned x_0, const unsigned y_0, const unsigned x_1, const unsigned y_1, const float sigma)
	{
		const unsigned diff_x = x_0 - x_1;
		const unsigned diff_y = y_0 - y_1;
		const float diff_square = diff_x * diff_x + diff_y * diff_y;
		const float normalized_diff_square = diff_square / (sigma * sigma);
		return expf(-0.5 * normalized_diff_square);
	}


	/**
	 * \brief Fill the unary energy given the input depth image.
	 *        Using 2x2 sub-sampling.
	 * \param depth_img uint16 texture mm-encoded depth image, original resolution
	 * \param unary_energy_map float2.x is for the energy for foreground, y is for backgound
	 */
	void computeUnaryForegoundSegmentation(
		cudaTextureObject_t depth_img,
		DeviceArray2D<float2> unary_energy_map,
		cudaStream_t stream = 0
	);


	/**
	 * \brief Compute the unary energy and the initial mean
	 *        field probability. The parameters are the same
	 *        as ones in unary computation method.
	 */
	void initMeanfieldUnaryForegroundSegmentation(
		cudaTextureObject_t raw_depth_img,
		cudaTextureObject_t clip_depth_img,
		DeviceArray2D<float2> unary_energy_map,
		cudaSurfaceObject_t meanfield_q,
		cudaStream_t stream = 0
	);


	/**
	 * \brief Fill the mask from the meanfield max-marginal, use 0.5 as threshold for binary
	 * \param meanfield_q The max marginal of each pixel to be foreground, float1 texture
	 * \param foreground_mask The result segmentation mask, uchar1 texture
	 */
	void writeForegroundSegmentationMask(
		cudaTextureObject_t meanfield_q,
		const unsigned rows, const unsigned cols,
		cudaSurfaceObject_t foreground_mask,
		cudaStream_t stream = 0
	);
}
