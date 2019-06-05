#include "common/common_types.h"
#include "common/global_configs.h"
#include "imgproc/segmentation/crf_common.h"
#include "imgproc/segmentation/crf_config.h"

#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	__global__ void computeUnaryForegoundSegmentationKernel(
		cudaTextureObject_t depth_img,
		PtrStepSz<float2> unary_energy_map //The first is for foreground, the next is for background
	){
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (y >= unary_energy_map.rows || x >= unary_energy_map.cols) return;
		
		
		//Perform a window search
		const auto origin_img_x = crf_subsample_rate * x;
		const auto origin_img_y = crf_subsample_rate * y;
		bool found = false;
		for(auto img_x = origin_img_x; img_x < origin_img_x + crf_subsample_rate; img_x++) {
			for(auto img_y = origin_img_y; img_y < origin_img_y + crf_subsample_rate; img_y++) {
				if(tex2D<unsigned short>(depth_img, img_x, img_y) > 0) {
					found = true;
				}
			}
		}

		//Assign to energy
		float2 unary_energy;
		if(found) {
			//If found, then this must be foreground
			unary_energy.x = 0.0f;
			unary_energy.y = 1e6f;
		} else {
			//If not found, not sure
			const float depth_notfound_confid_prob = 0.7f;
			unary_energy.x = - logf(1.0f - depth_notfound_confid_prob); //For foreground
			unary_energy.y = - logf(depth_notfound_confid_prob);
		}

		//Store to the map
		unary_energy_map.ptr(y)[x] = unary_energy;
	}

	__global__ void initMeanfieldUnaryForegroundSegmentationKernel(
		cudaTextureObject_t	raw_depth_img,
		cudaTextureObject_t clip_depth_img,
		PtrStepSz<float2> unary_energy_map, //The first is for foreground, the next is for background
		cudaSurfaceObject_t mean_field_foreground //The NORMALIZED foreground probability float texture, the negative is just 1 - tex2D<>(x, y)
	){
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (y >= unary_energy_map.rows || x >= unary_energy_map.cols) return;

		//Perform a window search
		const auto origin_clip_img_x = crf_subsample_rate * x;
		const auto origin_clip_img_y = crf_subsample_rate * y;
		bool clip_found = false;
		bool raw_found = false;
		for (auto img_x = origin_clip_img_x; img_x < origin_clip_img_x + crf_subsample_rate; img_x++) {
			for (auto img_y = origin_clip_img_y; img_y < origin_clip_img_y + crf_subsample_rate; img_y++) {
				if (tex2D<unsigned short>(clip_depth_img, img_x, img_y) > 0) {
					clip_found = true;
				}
				if (tex2D<unsigned short>(raw_depth_img, img_x + boundary_clip, img_y + boundary_clip) > 0) {
					raw_found = true;
				}
			}
		}

		//Assign to energy
		float2 unary_energy;
		float foreground_prob;

		const float confidence = 0.7;
		if (clip_found) { //This must be foreground
			//unary_energy.x = 0.0f;
			//unary_energy.y = 1e3f;
			//foreground_prob = 1.0f;
			foreground_prob = confidence;
			unary_energy.x = - __logf(foreground_prob);
			unary_energy.y = - __logf(1.0f - foreground_prob);
		}
		else if((!clip_found) && raw_found) { //This must be background
			//unary_energy.x = 1e3f;
			//unary_energy.y = 0.0f;
			//foreground_prob = 0.1f;
			foreground_prob = 1.0f - confidence;
			unary_energy.x = - __logf(foreground_prob);
			unary_energy.y = - __logf(1.0f - foreground_prob);
		} else { //Do not know
			unary_energy.x = - __logf(0.5f);
			unary_energy.y = unary_energy.x;
			foreground_prob = 0.5f;
		}

		//Store to the map and surface
		unary_energy_map.ptr(y)[x] = unary_energy;
		surf2Dwrite(foreground_prob, mean_field_foreground, x * sizeof(float), y);
	}

	__global__ void writeForegroundSegmentationMaskKernel(
		cudaTextureObject_t meanfield_q,
		const unsigned rows, const unsigned cols,
		cudaSurfaceObject_t foreground_mask
	) {
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x >= cols || y >= rows) return;

		//This value will finally be written to the surface
		unsigned char mask_value;
		const float meanfield_value = tex2D<float>(meanfield_q, x, y);
		if(meanfield_value > 0.5f) {
			mask_value = 1;
		} else {
			mask_value = 0;
		}

		//Write to surface
		surf2Dwrite(mask_value, foreground_mask, x * sizeof(unsigned char), y);
	}


}; /* End of namespace device */
}; /* End of namespace surfelwarp */


void surfelwarp::computeUnaryForegoundSegmentation(
	cudaTextureObject_t depth_img, 
	DeviceArray2D<float2> unary_energy_map,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(unary_energy_map.cols(), blk.x), divUp(unary_energy_map.rows(), blk.y));
	device::computeUnaryForegoundSegmentationKernel<<<grid, blk, 0, stream>>>(
		depth_img, 
		unary_energy_map
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::initMeanfieldUnaryForegroundSegmentation(
	cudaTextureObject_t raw_depth_img,
	cudaTextureObject_t clip_depth_img,
	DeviceArray2D<float2> unary_energy_map, 
	cudaSurfaceObject_t meanfield_q, 
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(unary_energy_map.cols(), blk.x), divUp(unary_energy_map.rows(), blk.y));
	device::initMeanfieldUnaryForegroundSegmentationKernel<<<grid, blk, 0, stream>>>(
		raw_depth_img,
		clip_depth_img,
		unary_energy_map, 
		meanfield_q
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::writeForegroundSegmentationMask(
	cudaTextureObject_t meanfield_q,
	const unsigned rows, const unsigned cols,
	cudaSurfaceObject_t foreground_mask,
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(cols, blk.x), divUp(rows, blk.y));
	device::writeForegroundSegmentationMaskKernel<<<grid, blk, 0, stream>>>(
		meanfield_q, 
		rows, cols, 
		foreground_mask
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}