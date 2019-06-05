#include <iostream>
#include <device_launch_parameters.h>
#include "common/sanity_check.h"
#include "common/common_utils.h"

__global__ void testTex1DFetchKernel(
	cudaTextureObject_t tex,
	const unsigned num_elems,
	surfelwarp::device::PtrSz<float> fromTexture
) {
	const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx < num_elems) {
		fromTexture[idx] = surfelwarp::fetch1DLinear<float>(tex, idx);
	}
}

int main()
{
	using namespace surfelwarp;

	//Prepare the test data on host
	const auto test_size = 1500;
	std::vector<float> h_vec;
	h_vec.resize(test_size);
	fillRandomVector(h_vec);

	//Upload it to device
	DeviceArray<float> d_vec;
	d_vec.upload(h_vec);

	//Create texture desc
	cudaTextureDesc texture_desc;
	memset(&texture_desc, 0, sizeof(cudaTextureDesc));
	texture_desc.normalizedCoords = 0;
	texture_desc.addressMode[0] = cudaAddressModeBorder;
	texture_desc.addressMode[1] = cudaAddressModeBorder;
	texture_desc.addressMode[2] = cudaAddressModeBorder;
	texture_desc.filterMode = cudaFilterModePoint;
	texture_desc.readMode = cudaReadModeElementType;
	texture_desc.sRGB = 0;

	//Create resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeLinear;
	resource_desc.res.linear.devPtr = d_vec.ptr();
	resource_desc.res.linear.sizeInBytes = d_vec.sizeBytes();
	resource_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resource_desc.res.linear.desc.x = 32;
	resource_desc.res.linear.desc.y = 0;
	resource_desc.res.linear.desc.z = 0;
	resource_desc.res.linear.desc.w = 0;

	//The texture object
	cudaTextureObject_t d_texture;
	cudaSafeCall(cudaCreateTextureObject(&d_texture, &resource_desc, &texture_desc, nullptr));

	//Access from kernel
	DeviceArray<float> fromTexture;
	fromTexture.create(test_size);
	dim3 blk(128);
	dim3 grid(divUp(test_size, blk.x));
	testTex1DFetchKernel<<<grid, blk>>>(d_texture, test_size, fromTexture);
	cudaDeviceSynchronize();

	//Compare the difference
	std::vector<float> h_from_text;
	fromTexture.download(h_from_text);
	auto err = maxRelativeError(h_from_text, h_vec);
	std::cout << "The err of text fetching iter 1 " << err << std::endl;

	//Check again with updated device array
	fillRandomVector(h_vec);
	d_vec.upload(h_vec);
	testTex1DFetchKernel<<<grid, blk>>>(d_texture, test_size, fromTexture);
	cudaDeviceSynchronize();
	
	//Download again
	fromTexture.download(h_from_text);
	err = maxRelativeError(h_from_text, h_vec);
	std::cout << "The err of text fetching iter 1 " << err << std::endl;
	return 0;
}