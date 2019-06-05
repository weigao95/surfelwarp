#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/surfel_types.h"
#include "common/ConfigParser.h"
#include "common/sanity_check.h"
#include "math/device_mat.h"
#include "math/DualQuaternion.hpp"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/DensityForegroundMapHandler.h"
#include "core/warp_solver/density_map_jacobian.cuh"
#include <device_launch_parameters.h>

/* Compute the gradient of density map and foreground mask
 */
namespace surfelwarp { namespace device {
	
	__global__ void computeDensityMapJacobian(
		cudaTextureObject_t reference_vertex_map,
		cudaTextureObject_t rendered_rgb_map,
		//The maps from camera observation
		cudaTextureObject_t density_map, // float1 texture
		cudaTextureObject_t density_gradient_map, // float2 texture
		unsigned width, unsigned height,
		//The queried pxiels and their weights
		const DeviceArrayView<ushort2> density_term_pixels,
		const ushort4* density_term_knn,
		const float4* density_term_knn_weight,
		//The warp field information
		const DualQuaternion* device_warp_field,
		const Intrinsic intrinsic, const mat34 world2camera,
		//Output
		TwistGradientOfScalarCost* gradient,
		float* residual_array
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx >= density_term_pixels.Size()) return; //The function does not involve warp/block sync

		//Prepare the data
		const ushort2 pixel = density_term_pixels[idx];
		const ushort4 knn = density_term_knn[idx];
		const float4 knn_weight = density_term_knn_weight[idx];
		const float4 reference_vertex = tex2D<float4>(reference_vertex_map, pixel.x, pixel.y);
		const float4 rendered_rgb = tex2D<float4>(rendered_rgb_map, pixel.x, pixel.y);
		const float geometry_density = rgb2density(rendered_rgb);
		
		//Compute the jacobian
		TwistGradientOfScalarCost twist_graident;
		float residual;
#if defined(USE_IMAGE_HUBER_WEIGHT)
		computeImageDensityJacobainAndResidualHuberWeight(
			density_map, 
			density_gradient_map, 
			width, height, 
			reference_vertex, 
			geometry_density, 
			knn, knn_weight, 
			device_warp_field, 
			intrinsic, world2camera, 
			twist_graident, residual,
			d_density_map_cutoff
		);
#else
		computeImageDensityJacobainAndResidual(
			density_map, 
			density_gradient_map, 
			width, height, 
			reference_vertex, 
			geometry_density, 
			knn, knn_weight, 
			device_warp_field, 
			intrinsic, world2camera, 
			twist_graident, residual
		);
#endif

		//Output
		gradient[idx] = twist_graident;
		residual_array[idx] = residual;
	}


	__global__ void computeForegroundMaskJacobian(
		cudaTextureObject_t reference_vertex_map,
		//The maps from camera observation
		cudaTextureObject_t filter_foreground_mask, // float1 texture
		cudaTextureObject_t foreground_gradient_map, // float2 texture
		unsigned width, unsigned height,
		//The queried pxiels and their weights
		const DeviceArrayView<ushort2> foreground_term_pixels,
		const ushort4* foreground_term_knn,
		const float4* foreground_term_knn_weight,
		//The warp field information
		const DualQuaternion* device_warp_field,
		const Intrinsic intrinsic, const mat34 world2camera,
		//Output
		TwistGradientOfScalarCost* gradient,
		float* residual_array
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx >= foreground_term_pixels.Size()) return; //The function does not involve warp/block sync

		//Prepare the data
		const ushort2 pixel = foreground_term_pixels[idx];
		const ushort4 knn = foreground_term_knn[idx];
		const float4 knn_weight = foreground_term_knn_weight[idx];
		const float4 reference_vertex = tex2D<float4>(reference_vertex_map, pixel.x, pixel.y);
		const float geometry_density = 0.0f; // An occupied pixel should be marked as 0 on filter foreground mask
		
		//Compute the jacobian
		TwistGradientOfScalarCost twist_graident;
		float residual;
#if defined(USE_IMAGE_HUBER_WEIGHT)
		computeImageDensityJacobainAndResidualHuberWeight(
			filter_foreground_mask, 
			foreground_gradient_map, 
			width, height, 
			reference_vertex, 
			geometry_density, 
			knn, knn_weight, 
			device_warp_field, 
			intrinsic, world2camera, 
			twist_graident, residual,
			d_foreground_cutoff
		);
#else
		computeImageDensityJacobainAndResidual(
			filter_foreground_mask, 
			foreground_gradient_map, 
			width, height, 
			reference_vertex, 
			geometry_density, 
			knn, knn_weight, 
			device_warp_field, 
			intrinsic, world2camera, 
			twist_graident, residual
		);
#endif

		//Output
		gradient[idx] = twist_graident;
		residual_array[idx] = residual;
	}


} // namespace device	
} // namespace surfelwarp


void surfelwarp::DensityForegroundMapHandler::computeDensityMapTwistGradient(cudaStream_t stream)
{
	//Correct the size of output
	const auto num_pixels = m_potential_pixels_knn.pixels.Size();
	m_color_residual.ResizeArrayOrException(num_pixels);
	m_color_twist_gradient.ResizeArrayOrException(num_pixels);

	//If the size is zero, just return
	if (num_pixels == 0) {
		return;
	}
	
	dim3 blk(128);
	dim3 grid(divUp(num_pixels, blk.x));
	device::computeDensityMapJacobian<<<grid, blk, 0, stream>>>(
		m_geometry_maps.reference_vertex_map,
		m_geometry_maps.normalized_rgb_map,
		m_depth_observation.density_map,
		m_depth_observation.density_gradient_map,
		m_image_width, m_image_height,
		m_potential_pixels_knn.pixels,
		m_potential_pixels_knn.node_knn.RawPtr(),
		m_potential_pixels_knn.knn_weight.RawPtr(),
		m_node_se3,
		m_project_intrinsic, m_world2camera,
		m_color_twist_gradient.Ptr(),
		m_color_residual.Ptr()
	);

#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::DensityForegroundMapHandler::computeForegroundMaskTwistGradient(cudaStream_t stream)
{
	//Correct the size of output
	m_foreground_residual.ResizeArrayOrException(m_valid_mask_pixel.ArraySize());
	m_foreground_twist_gradient.ResizeArrayOrException(m_valid_mask_pixel.ArraySize());
	
	//If the size is zero, just return
	if(m_valid_mask_pixel.ArraySize() == 0) {
		LOG(INFO) << "There is no foreground pixel";
		return;
	}
	
	//Invoke the kernel
	dim3 blk(64);
	dim3 grid(divUp(m_valid_mask_pixel.ArraySize(), blk.x));
	device::computeForegroundMaskJacobian<<<grid, blk, 0, stream>>>(
		m_geometry_maps.reference_vertex_map,
		m_depth_observation.filtered_foreground_mask,
		m_depth_observation.foreground_mask_gradient_map,
		m_image_width, m_image_height,
		m_valid_mask_pixel.ArrayView(),
		m_valid_mask_pixel_knn.Ptr(),
		m_valid_mask_pixel_knn_weight.Ptr(),
		m_node_se3,
		m_project_intrinsic, m_world2camera,
		m_foreground_twist_gradient.Ptr(),
		m_foreground_residual.Ptr()
	);

#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::DensityForegroundMapHandler::ComputeTwistGradient(
	cudaStream_t color_stream,
	cudaStream_t foreground_stream
) {
	computeDensityMapTwistGradient(color_stream);
	computeForegroundMaskTwistGradient(foreground_stream);
}


void surfelwarp::DensityForegroundMapHandler::Term2JacobianMaps(
	DensityMapTerm2Jacobian &density_term2jacobian,
	ForegroundMaskTerm2Jacobian &foreground_term2jacobian
) {
	density_term2jacobian.knn_array = m_potential_pixels_knn.node_knn;
	density_term2jacobian.knn_weight_array = m_potential_pixels_knn.knn_weight;
	density_term2jacobian.residual_array = m_color_residual.ArrayView();
	density_term2jacobian.twist_gradient_array = m_color_twist_gradient.ArrayView();
	density_term2jacobian.check_size();
	
	foreground_term2jacobian.knn_array = m_valid_mask_pixel_knn.ArrayView();
	foreground_term2jacobian.knn_weight_array = m_valid_mask_pixel_knn_weight.ArrayView();
	foreground_term2jacobian.residual_array = m_foreground_residual.ArrayView();
	foreground_term2jacobian.twist_gradient_array = m_foreground_twist_gradient.ArrayView();
	foreground_term2jacobian.check_size();
}