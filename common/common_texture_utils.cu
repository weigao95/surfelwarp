#include "common/common_texture_utils.h"

cudaTextureObject_t surfelwarp::create1DLinearTexture(const DeviceArray<float> &array) {
	cudaTextureDesc texture_desc;
	memset(&texture_desc, 0, sizeof(cudaTextureDesc));
	texture_desc.normalizedCoords = 0;
	texture_desc.addressMode[0] = cudaAddressModeBorder; //Return 0 outside the boundary
	texture_desc.addressMode[1] = cudaAddressModeBorder;
	texture_desc.addressMode[2] = cudaAddressModeBorder;
	texture_desc.filterMode = cudaFilterModePoint;
	texture_desc.readMode = cudaReadModeElementType;
	texture_desc.sRGB = 0;

	//Create resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeLinear;
	resource_desc.res.linear.devPtr = (void*)array.ptr();
	resource_desc.res.linear.sizeInBytes = array.sizeBytes();
	resource_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resource_desc.res.linear.desc.x = 32;
	resource_desc.res.linear.desc.y = 0;
	resource_desc.res.linear.desc.z = 0;
	resource_desc.res.linear.desc.w = 0;

	//Allocate the texture
	cudaTextureObject_t d_texture;
	cudaSafeCall(cudaCreateTextureObject(&d_texture, &resource_desc, &texture_desc, nullptr));
	return d_texture;
}

cudaTextureObject_t surfelwarp::create1DLinearTexture(const DeviceBufferArray<float>& array) {
	DeviceArray<float> pcl_array((float*)array.Ptr(), array.Capacity());
	return create1DLinearTexture(pcl_array);
}

void surfelwarp::createDefault2DTextureDesc(cudaTextureDesc &desc) {
	memset(&desc, 0, sizeof(desc));
	desc.addressMode[0] = cudaAddressModeBorder; //Return 0 outside the boundary
	desc.addressMode[1] = cudaAddressModeBorder;
	desc.addressMode[2] = cudaAddressModeBorder;
	desc.filterMode = cudaFilterModePoint;
	desc.readMode = cudaReadModeElementType;
	desc.normalizedCoords = 0;
}


void surfelwarp::createDepthTexture(
	const unsigned img_rows,
	const unsigned img_cols,
	cudaTextureObject_t &texture,
	cudaArray_t &d_array
) {
	//The texture description
	cudaTextureDesc depth_texture_desc;
	createDefault2DTextureDesc(depth_texture_desc);

	//Create channel descriptions
	cudaChannelFormatDesc depth_channel_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);

	//Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &depth_channel_desc, img_cols, img_rows));

	//Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	//Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &depth_texture_desc, 0));
}



void surfelwarp::createDepthTextureSurface(
	const unsigned img_rows,
	const unsigned img_cols,
	cudaTextureObject_t &texture,
	cudaSurfaceObject_t &surface,
	cudaArray_t &d_array
) {
	//The texture description
	cudaTextureDesc depth_texture_desc;
	createDefault2DTextureDesc(depth_texture_desc);

	//Create channel descriptions
	cudaChannelFormatDesc depth_channel_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);

	//Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &depth_channel_desc, img_cols, img_rows));

	//Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	//Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &depth_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}


void surfelwarp::createDepthTextureSurface(const unsigned img_rows, const unsigned img_cols, CudaTextureSurface & collect) {
	createDepthTextureSurface(
		img_rows, img_cols,
		collect.texture, collect.surface, collect.d_array);
}



void surfelwarp::createFloat4TextureSurface(
	const unsigned rows, const unsigned cols,
	cudaTextureObject_t &texture,
	cudaSurfaceObject_t &surface,
	cudaArray_t &d_array
) {
	//The texture description
	cudaTextureDesc float4_texture_desc;
	createDefault2DTextureDesc(float4_texture_desc);

	//Create channel descriptions
	cudaChannelFormatDesc float4_channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	//Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &float4_channel_desc, cols, rows));

	//Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	//Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &float4_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}


void surfelwarp::createFloat4TextureSurface(const unsigned rows, const unsigned cols, CudaTextureSurface & texture_collect)
{
	createFloat4TextureSurface(
		rows, cols,
		texture_collect.texture,
		texture_collect.surface,
		texture_collect.d_array
	);
}


void surfelwarp::createFloat1TextureSurface(
	const unsigned rows, const unsigned cols, 
	cudaTextureObject_t & texture, 
	cudaSurfaceObject_t & surface, 
	cudaArray_t & d_array
) {
	//The texture description
	cudaTextureDesc float1_texture_desc;
	createDefault2DTextureDesc(float1_texture_desc);

	//Create channel descriptions
	cudaChannelFormatDesc float1_channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	//Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &float1_channel_desc, cols, rows));

	//Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	//Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &float1_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}


void surfelwarp::createFloat1TextureSurface(
	const unsigned rows, const unsigned cols, 
	CudaTextureSurface & texture_collect
) {
	createFloat1TextureSurface(
		rows, cols, 
		texture_collect.texture, 
		texture_collect.surface, 
		texture_collect.d_array
	);
}


void surfelwarp::createFloat2TextureSurface(
	const unsigned rows, const unsigned cols, 
	cudaTextureObject_t & texture, 
	cudaSurfaceObject_t & surface, 
	cudaArray_t & d_array
) {
	//The texture description
	cudaTextureDesc float2_texture_desc;
	createDefault2DTextureDesc(float2_texture_desc);

	//Create channel descriptions
	cudaChannelFormatDesc float2_channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

	//Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &float2_channel_desc, cols, rows));

	//Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	//Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &float2_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void surfelwarp::createFloat2TextureSurface(
	const unsigned rows, const unsigned cols, 
	CudaTextureSurface & texture_collect
) {
	createFloat2TextureSurface(
		rows, cols, 
		texture_collect.texture, 
		texture_collect.surface, 
		texture_collect.d_array
	);
}


void surfelwarp::createUChar1TextureSurface(
	const unsigned rows, const unsigned cols, 
	cudaTextureObject_t & texture, 
	cudaSurfaceObject_t & surface,
	cudaArray_t & d_array
) {
	//The texture description
	cudaTextureDesc uchar1_texture_desc;
	createDefault2DTextureDesc(uchar1_texture_desc);

	//Create channel descriptions
	cudaChannelFormatDesc uchar1_channel_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

	//Allocate the cuda array
	cudaSafeCall(cudaMallocArray(&d_array, &uchar1_channel_desc, cols, rows));

	//Create the resource desc
	cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = d_array;

	//Allocate the texture
	cudaSafeCall(cudaCreateTextureObject(&texture, &resource_desc, &uchar1_texture_desc, 0));
	cudaSafeCall(cudaCreateSurfaceObject(&surface, &resource_desc));
}

void surfelwarp::createUChar1TextureSurface(
	const unsigned rows, const unsigned cols, 
	CudaTextureSurface & texture_collect
) {
	createUChar1TextureSurface(
		rows, cols, 
		texture_collect.texture, 
		texture_collect.surface, 
		texture_collect.d_array
	);
}



void surfelwarp::query2DTextureExtent(cudaTextureObject_t texture, unsigned &width, unsigned &height) {
	cudaResourceDesc texture_res;
	cudaSafeCall(cudaGetTextureObjectResourceDesc(&texture_res, texture));
	cudaArray_t cu_array = texture_res.res.array.array;
	cudaChannelFormatDesc channel_desc;
	cudaExtent extent;
	unsigned int flag;
	cudaSafeCall(cudaArrayGetInfo(&channel_desc, &extent, &flag, cu_array));

	width = extent.width;
	height = extent.height;
}

void surfelwarp::releaseTextureCollect(CudaTextureSurface & texture_collect)
{
	cudaSafeCall(cudaDestroyTextureObject(texture_collect.texture));
	cudaSafeCall(cudaDestroySurfaceObject(texture_collect.surface));
	cudaSafeCall(cudaFreeArray(texture_collect.d_array));
}

