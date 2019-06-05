#include "common/common_texture_utils.h"
#include "visualization/Visualizer.h"

#include <device_launch_parameters.h>

#include <opencv2/opencv.hpp>

namespace surfelwarp { namespace device {

	__global__ void markValidIndexMapPixelKernel(
		cudaTextureObject_t index_map,
		int validity_halfsize,
		unsigned img_rows, unsigned img_cols,
		unsigned char* flatten_validity_indicator
	) {
		const auto x_center = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y_center = threadIdx.y + blockDim.y * blockIdx.y;
		if(x_center >= img_cols || y_center >= img_rows) return;
		const auto offset = x_center + y_center * img_cols;

		//Only depend on this pixel
		if(validity_halfsize <= 0) {
			const auto surfel_index = tex2D<unsigned>(index_map, x_center, y_center);
			unsigned char validity = 0;
			if(surfel_index != 0xFFFFFFFF) validity = 1;

			//Write it and return
			flatten_validity_indicator[offset] = validity;
			return;
		}

		//Should perform a window search as the halfsize is at least 1
		unsigned char validity = 1;
		for(auto y = y_center - validity_halfsize; y <= y_center + validity_halfsize; y++) {
			for(auto x = x_center - validity_halfsize; x <= x_center + validity_halfsize; x++) {
				if(tex2D<unsigned>(index_map, x, y) == 0xFFFFFFFF) validity = 0;
			}
		}

		//Save it
		flatten_validity_indicator[offset] = validity;
	}


} // device
} // surfelwarp

void surfelwarp::Visualizer::DrawValidIndexMap(cudaTextureObject_t index_map, int validity_halfsize) {
	//Query the map
	cv::Mat validity_map = GetValidityMapCV(index_map, validity_halfsize);
	
	//Draw it
	DrawRGBImage(validity_map);
}


void surfelwarp::Visualizer::SaveValidIndexMap(
	cudaTextureObject_t index_map,
	int validity_halfsize,
	const std::string &path
) {
	//Query the map
	cv::Mat validity_map = GetValidityMapCV(index_map, validity_halfsize);
	
	//Save it
	cv::imwrite(path, validity_map);
}

cv::Mat surfelwarp::Visualizer::GetValidityMapCV(cudaTextureObject_t index_map, int validity_halfsize) {
	//Query the size
	unsigned width, height;
	query2DTextureExtent(index_map, width, height);
	
	//Malloc
	DeviceArray<unsigned char> flatten_validity_indicator;
	flatten_validity_indicator.create(width * height);
	
	//Mark the validity
	MarkValidIndexMapValue(index_map, validity_halfsize, flatten_validity_indicator);
	
	//Download it and transfer it into cv::Mat
	std::vector<unsigned char> h_validity_array;
	flatten_validity_indicator.download(h_validity_array);
	
	//The validity map
	cv::Mat validity_map = cv::Mat(height, width, CV_8UC1);
	unsigned num_valid_pixel = 0;
	for(auto y = 0; y < height; y++) {
		for(auto x = 0; x < width; x++) {
			const auto offset = x + y * width;
			if(h_validity_array[offset] > 0) {
				validity_map.at<unsigned char>(y, x) = 255;
				num_valid_pixel++;
			} else {
				validity_map.at<unsigned char>(y, x) = 0;
			}
		}
	}
	
	//Log the number of valid pixel
	//LOG(INFO) << "The number of valid pixel in the index map of rendered geometry with validity halfsize " << validity_halfsize << " is " << num_valid_pixel;
	return validity_map;
}

void surfelwarp::Visualizer::MarkValidIndexMapValue(
	cudaTextureObject_t index_map,
	int validity_halfsize,
	surfelwarp::DeviceArray<unsigned char> flatten_validity_indicator
) {
	//Query the size
	unsigned width, height;
	query2DTextureExtent(index_map, width, height);

	//Do it
	dim3 blk(16, 16);
	dim3 grid(divUp(width, blk.x), divUp(height, blk.y));
	device::markValidIndexMapPixelKernel<<<grid, blk>>>(
		index_map, 
		validity_halfsize, 
		height, width, 
		flatten_validity_indicator.ptr()
	);

	//Always sync and check error
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
}



