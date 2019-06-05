#pragma once

#include "common/common_types.h"
#include "common/DeviceBufferArray.h"
#include <cuda_texture_types.h>
#include <exception>
#include <cuda.h>

namespace surfelwarp
{
	
	/**
	* \brief Create of 1d linear float texturem, accessed by fetch1DLinear.
	*        Using the array as the underline memory
	*/
	cudaTextureObject_t create1DLinearTexture(const DeviceArray<float>& array);
	cudaTextureObject_t create1DLinearTexture(const DeviceBufferArray<float>& array);


	/**
	* \brief Create TextureDesc for default 2D texture
	*/
	void createDefault2DTextureDesc(cudaTextureDesc& desc);



	/**
	* \brief Create 2D uint16 textures (and surfaces) for depth image
	*/
	void createDepthTexture(
		const unsigned img_rows, const unsigned img_cols,
		cudaTextureObject_t& texture, cudaArray_t& d_array
	);
	void createDepthTextureSurface(
		const unsigned img_rows, const unsigned img_cols,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array
	);
	void createDepthTextureSurface(
		const unsigned img_rows, const unsigned img_cols,
		CudaTextureSurface& collect
	);


	/**
	* \brief Create 2D float4 textures (and surfaces) for all kinds of use
	*/
	void createFloat4TextureSurface(
		const unsigned rows, const unsigned cols,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array
	);
	void createFloat4TextureSurface(
		const unsigned rows, const unsigned cols,
		CudaTextureSurface& texture_collect
	);



	/**
	* \brief Create 2D float1 textures (and surfaces) for mean-field inference
	*/
	void createFloat1TextureSurface(
		const unsigned rows, const unsigned cols,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array
	);
	void createFloat1TextureSurface(
		const unsigned rows, const unsigned cols,
		CudaTextureSurface& texture_collect
	);



	/**
	* \brief Create 2D float2 textures (and surfaces) for gradient map
	*/
	void createFloat2TextureSurface(
		const unsigned rows, const unsigned cols,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array
	);
	void createFloat2TextureSurface(
		const unsigned rows, const unsigned cols,
		CudaTextureSurface& texture_collect
	);



	/**
	* \brief Create 2D uchar1 textures (and surfaces) for binary mask
	*/
	void createUChar1TextureSurface(
		const unsigned rows, const unsigned cols,
		cudaTextureObject_t& texture, cudaSurfaceObject_t& surface,
		cudaArray_t& d_array
	);
	void createUChar1TextureSurface(
		const unsigned rows, const unsigned cols,
		CudaTextureSurface& texture_collect
	);



	/**
	* \brief Release 2D texture
	*/
	void releaseTextureCollect(CudaTextureSurface& texture_collect);

	/**
	* \brief The query functions for 2D texture
	*/
	void query2DTextureExtent(cudaTextureObject_t texture, unsigned& width, unsigned& height);
	
	
	
} // namespace surfelwarp