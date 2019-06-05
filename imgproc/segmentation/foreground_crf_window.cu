#include "math/vector_ops.hpp"
#include "imgproc/segmentation/crf_config.h"
#include "imgproc/segmentation/foreground_crf_window.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__global__ void foregroundMeanfieldIterWindowKernel(
		//The input maps
		cudaTextureObject_t meanfield_foreground_in,
		cudaTextureObject_t rgb_img,
		PtrStepSz<const float2> unary_energy_map,
		//Normalizing constants
		const float sigma_alpha, 
		const float sigma_beta,
		const float sigma_gamma,
		//The weight constants
		const float appearance_weight,
		const float smooth_weight,
		//Output
		cudaSurfaceObject_t meanfield_foreground_out
	) {
		const int x = threadIdx.x + blockDim.x * blockIdx.x;
		const int y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x >= unary_energy_map.cols || y >= unary_energy_map.rows) return;

		//Read the pixel
		const int rgb_x = crf_subsample_rate * x;
		const int rgb_y = crf_subsample_rate * y;
		float4 normalized_rgba = tex2D<float4>(rgb_img, rgb_x, rgb_y);

		//Construct the feature for this pixel
		float feature_0[5], feature_1[5];
		feature_0[0] = float(x) / sigma_alpha;
		feature_0[1] = float(y) / sigma_alpha;
		feature_0[2] = normalized_rgba.x * 255.f / sigma_beta;
		feature_0[3] = normalized_rgba.y * 255.f / sigma_beta;
		feature_0[4] = normalized_rgba.z * 255.f / sigma_beta;

		//Perform window search
		const unsigned halfsize = 7;
		float e_foreground = 0.0f, e_background = 0.0f;
		for(int neighbor_y = y - halfsize; neighbor_y <= y + halfsize; neighbor_y++) {
			for(int neighbor_x = x - halfsize; neighbor_x <= x + halfsize; neighbor_x++) {
				//Construct the feature
				const auto neighbour_rgb_x = crf_subsample_rate * neighbor_x;
				const auto neighbour_rgb_y = crf_subsample_rate * neighbor_y;
				normalized_rgba = tex2D<float4>(rgb_img, neighbour_rgb_x, neighbour_rgb_y);
				feature_1[0] = float(neighbor_x) / sigma_alpha;
				feature_1[1] = float(neighbor_y) / sigma_alpha;
				feature_1[2] = normalized_rgba.x * 255.f / sigma_beta;
				feature_1[3] = normalized_rgba.y * 255.f / sigma_beta;
				feature_1[4] = normalized_rgba.z * 255.f / sigma_beta;

				//Compute the kernel value
				float kernel_value = 0.0;
				kernel_value += appearance_weight * appearance_kernel(feature_0, feature_1);
				kernel_value += smooth_weight * smooth_kernel(x, y, neighbor_x, neighbor_y, sigma_gamma);
				
				//Message passing
				const float neighbor_foreground_prob = tex2D<float>(meanfield_foreground_in, neighbor_x, neighbor_y);
				const float neighbor_backround_prob = 1.0f - neighbor_foreground_prob;

				//Note that the window might be outside, in that case, tex2D should return zero
				if(neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < unary_energy_map.cols && neighbor_y < unary_energy_map.rows) {
					e_foreground += (neighbor_backround_prob * kernel_value);
					e_background += (neighbor_foreground_prob * kernel_value);
				}
			}
		}

		//Subtract the contribution of this pixel
		//If the feature perfectly match, the kernel value is 1.0f
		const float prev_foreground_prob = tex2D<float>(meanfield_foreground_in, x, y);
		const float prev_backround_prob = 1.0f - prev_foreground_prob;
		e_foreground -= prev_backround_prob * (appearance_weight + smooth_weight);
		e_background -= prev_foreground_prob * (appearance_weight + smooth_weight);

		//Update the mean field locally
		const float2 unary_energy = unary_energy_map.ptr(y)[x];
		const float foreground_energy = unary_energy.x + e_foreground;
		const float background_energy = unary_energy.y + e_background;
		const float energy_diff = foreground_energy - background_energy;

		//Note the numerical problem involved with expf
		float foreground_prob;
		const float energy_cutoff = 20;
		if(energy_diff < - energy_cutoff) {
			foreground_prob = 1.0f;
		} else if(energy_diff > energy_cutoff) {
			foreground_prob = 0.0f;
		} else {
			const float exp_energy_diff = __expf(energy_diff);
			foreground_prob = 1.0f / (1.0f + exp_energy_diff);
		}
		

		//Will there might be numerical errors
		if (foreground_prob > 1.0f) {
			foreground_prob = 1.0f;
		} else if(foreground_prob < 0.0f) {
			foreground_prob = 1e-3f;
		}

		//Write to the surface
		surf2Dwrite(foreground_prob, meanfield_foreground_out, x * sizeof(float), y);
	}

}; /* End of namespace device */
}; /* End of namespace surfelwarp */



void surfelwarp::foregroundMeanfieldIterWindow(
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
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(unary_energy_map.cols(), blk.x), divUp(unary_energy_map.rows(), blk.y));
	device::foregroundMeanfieldIterWindowKernel<<<grid, blk, 0, stream>>>(
		meanfield_foreground_in, 
		rgb_img, 
		unary_energy_map,
		sigma_alpha, 
		sigma_beta, 
		sigma_gamma, 
		appearance_weight,
		smooth_weight, 
		meanfield_foregound_out
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}