#include "core/warp_solver/ImageTermKNNFetcher.h"
#include "core/warp_solver/solver_constants.h"

#include <device_launch_parameters.h>


namespace surfelwarp { namespace device {
	
	//Only mark the pixel that corresponded to valid input
	//Let the term handler to deal with other issues
	__global__ void markPotentialValidImageTermPixelKernel(
		cudaTextureObject_t index_map,
		unsigned img_rows, unsigned img_cols,
		unsigned* reference_pixel_indicator
	) {
		const auto x = threadIdx.x + blockDim.x*blockIdx.x;
		const auto y = threadIdx.y + blockDim.y*blockIdx.y;
		if (x >= img_cols || y >= img_rows) return;

		//The indicator will must be written to pixel_occupied_array
		const auto offset = y * img_cols + x;

		//Read the value on index map
		const auto surfel_index = tex2D<unsigned>(index_map, x, y);

		//Need other criterion?
		unsigned indicator = 0;
		if (surfel_index != d_invalid_index) {
			indicator = 1;
		}

		reference_pixel_indicator[offset] = indicator;
	}

	__global__ void compactPontentialImageTermPixelsKernel(
		const DeviceArrayView2D<KNNAndWeight> knn_map,
		const unsigned* potential_pixel_indicator,
		const unsigned* prefixsum_pixel_indicator,
		ushort2* potential_pixels,
		ushort4* potential_pixels_knn,
		float4*  potential_pixels_knn_weight
	) {
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if (x >= knn_map.Cols() || y >= knn_map.Rows()) return;
		const auto flatten_idx = x + y * knn_map.Cols();
		if (potential_pixel_indicator[flatten_idx] > 0)
		{
			const auto offset = prefixsum_pixel_indicator[flatten_idx] - 1;
			const KNNAndWeight knn = knn_map(y, x);
			potential_pixels[offset] = make_ushort2(x, y);
			potential_pixels_knn[offset] = knn.knn;
			potential_pixels_knn_weight[offset] = knn.weight;
		}
	}


} // device
} // surfelwarp


void surfelwarp::ImageTermKNNFetcher::MarkPotentialMatchedPixels(cudaStream_t stream) {
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_width, blk.x), divUp(m_image_height, blk.y));
	device::markPotentialValidImageTermPixelKernel <<<grid, blk, 0, stream>>>(
		m_geometry_maps.index_map,
		m_image_height,
		m_image_width,
		m_potential_pixel_indicator.ptr()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::ImageTermKNNFetcher::CompactPotentialValidPixels(cudaStream_t stream) {
	//Do a prefix sum
	m_indicator_prefixsum.InclusiveSum(m_potential_pixel_indicator, stream);

	//Invoke the kernel
	dim3 blk(16, 16);
	dim3 grid(divUp(m_image_width, blk.x), divUp(m_image_height, blk.y));
	device::compactPontentialImageTermPixelsKernel <<<grid, blk, 0, stream>>>(
		m_geometry_maps.knn_map,
		m_potential_pixel_indicator,
		m_indicator_prefixsum.valid_prefixsum_array,
		m_potential_pixels.Ptr(),
		m_dense_image_knn.Ptr(),
		m_dense_image_knn_weight.Ptr()
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::ImageTermKNNFetcher::SyncQueryCompactedPotentialPixelSize(cudaStream_t stream) {
	//unsigned num_potential_pairs;
	cudaSafeCall(cudaMemcpyAsync(
		m_num_potential_pixel,
		m_indicator_prefixsum.valid_prefixsum_array.ptr() + m_potential_pixel_indicator.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream
	));
	cudaSafeCall(cudaStreamSynchronize(stream));

	m_potential_pixels.ResizeArrayOrException(*m_num_potential_pixel);
	m_dense_image_knn.ResizeArrayOrException(*m_num_potential_pixel);
	m_dense_image_knn_weight.ResizeArrayOrException(*m_num_potential_pixel);
}