#pragma once

#include "imgproc/segmentation/crf_config.h"
#include "imgproc/segmentation/crf_common.h"

namespace surfelwarp {
	

	/**
	 * \brief Compute one iterations of mean-field CRF using brute-force
	 *        windowsed search. 
	 * \param meanfield_foreground_in float1 texture at subsampled resolution as input
	 * \param rgb_img float4 [-1, 1] rgb image texture at original resolution
	 * \param unary_energy_map The unary energy map at subsampled resolution.
	 *                         float2.x is for foreground, float2.y is for background
	 * \param sigma_alpha Refer to Philipp Krahenbuhl et al "Efficient Inference in Fully Connected CRFs with 
	 *                    Gaussian Edge Potentials" for these constants
	 * \param meanfield_foregound_out The output meanfield MUST NOT be the same as input
	 */
	void foregroundMeanfieldIterWindow(
		//The input maps
		cudaTextureObject_t meanfield_foreground_in,
		cudaTextureObject_t rgb_img,
		const DeviceArray2D<float2>& unary_energy_map,
		//Normalizing constants
		const float sigma_alpha,
		const float sigma_beta,
		const float sigma_gamma,
		//The weight functions
		const float appearance_weight,
		const float smooth_weight,
		//Output
		cudaSurfaceObject_t meanfield_foregound_out,
		cudaStream_t stream
	);
}