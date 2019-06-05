//
// Created by wei on 3/31/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/DeviceBufferArray.h"
#include "common/surfel_types.h"
#include "common/algorithm_types.h"
#include "math/DualQuaternion.hpp"
#include "core/warp_solver/solver_types.h"
#include "core/warp_solver/ImageTermKNNFetcher.h"
#include <memory>

namespace surfelwarp {
	
	class DensityForegroundMapHandler {
	private:
		//The info from config
		int m_image_height;
		int m_image_width;
		Intrinsic m_project_intrinsic;
		
		//The info from solver
		DeviceArrayView<DualQuaternion> m_node_se3;
		DeviceArrayView2D<KNNAndWeight> m_knn_map;
		mat34 m_world2camera;
		
		//The maps from depth observation
		struct {
			cudaTextureObject_t foreground_mask; // uchar texture
			cudaTextureObject_t filtered_foreground_mask; // float1 texture
			cudaTextureObject_t foreground_mask_gradient_map; // float2 texture

			//The density map from depth
			cudaTextureObject_t density_map; // float1 texture
			cudaTextureObject_t density_gradient_map; // float2 texture
		} m_depth_observation;
		

		//The map from renderer
		struct {
			cudaTextureObject_t reference_vertex_map;
			cudaTextureObject_t reference_normal_map;
			cudaTextureObject_t index_map;
			cudaTextureObject_t normalized_rgb_map;
		} m_geometry_maps;
		
		//The pixel from the indexer
		ImageTermKNNFetcher::ImageTermPixelAndKNN m_potential_pixels_knn;
	
	public:
		using Ptr = std::shared_ptr<DensityForegroundMapHandler>;
		DensityForegroundMapHandler();
		~DensityForegroundMapHandler() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(DensityForegroundMapHandler);

		//Explicit allocate
		void AllocateBuffer();
		void ReleaseBuffer();
		

		//Set input
		void SetInputs(
			const DeviceArrayView<DualQuaternion>& node_se3,
			const DeviceArrayView2D<KNNAndWeight>& knn_map,
			//The foreground mask terms
			cudaTextureObject_t foreground_mask, 
			cudaTextureObject_t filtered_foreground_mask,
			cudaTextureObject_t foreground_gradient_map,
			//The color density terms
			cudaTextureObject_t density_map,
			cudaTextureObject_t density_gradient_map,
			//The rendered maps
			cudaTextureObject_t reference_vertex_map,
			cudaTextureObject_t reference_normal_map,
			cudaTextureObject_t index_map,
			cudaTextureObject_t normalized_rgb_map,
			const mat34& world2camera,
			//The potential pixels,
			const ImageTermKNNFetcher::ImageTermPixelAndKNN& potential_pixels_knn
		);
		
		//Update the node se3
		void UpdateNodeSE3(DeviceArrayView<DualQuaternion> node_se3);
		

		//The finder interface
		void FindValidColorForegroundMaskPixels(cudaStream_t color_stream = 0, cudaStream_t mask_stream = 0);
		void FindPotentialForegroundMaskPixelSynced(cudaStream_t stream = 0);
		

		/* Mark the valid pixel for both color and foreground mask
		 */
	private:
		//These should be 2D maps
		DeviceArray<unsigned> m_color_pixel_indicator_map;
		DeviceArray<unsigned> m_mask_pixel_indicator_map;
	public:
		void MarkValidColorForegroundMaskPixels(cudaStream_t stream = 0);
		
		
		/* The compaction for maps
		 */
	private:
		PrefixSum m_color_pixel_indicator_prefixsum;
		PrefixSum m_mask_pixel_indicator_prefixsum;
		DeviceBufferArray<ushort2> m_valid_color_pixel, m_valid_mask_pixel;
		DeviceBufferArray<ushort4> m_valid_color_pixel_knn, m_valid_mask_pixel_knn;
		DeviceBufferArray<float4> m_valid_color_pixel_knn_weight, m_valid_mask_pixel_knn_weight;

		//The pagelocked memory
		unsigned* m_num_mask_pixel;
	public:
		void CompactValidColorPixel(cudaStream_t stream = 0);
		void QueryCompactedColorPixelArraySize(cudaStream_t stream = 0);
		void CompactValidMaskPixel(cudaStream_t stream = 0);
		void QueryCompactedMaskPixelArraySize(cudaStream_t stream = 0);
		
		
		/* Compute the gradient
		 */
	private:
		DeviceBufferArray<float> m_color_residual;
		DeviceBufferArray<TwistGradientOfScalarCost> m_color_twist_gradient;
		DeviceBufferArray<float> m_foreground_residual;
		DeviceBufferArray<TwistGradientOfScalarCost> m_foreground_twist_gradient;
		void computeDensityMapTwistGradient(cudaStream_t stream = 0);
		void computeForegroundMaskTwistGradient(cudaStream_t stream = 0);
	public:
		void ComputeTwistGradient(cudaStream_t color_stream, cudaStream_t foreground_stream);
		void Term2JacobianMaps(
			DensityMapTerm2Jacobian& density_term2jacobian,
			ForegroundMaskTerm2Jacobian& foreground_term2jacobian
		);

		
		/* The access interface
		 */
	public:
		DeviceArrayView<ushort4> DensityTermKNN() const { return m_valid_color_pixel_knn.ArrayView(); }
		DeviceArrayView<ushort4> ForegroundMaskTermKNN() const { return m_valid_mask_pixel_knn.ArrayView(); }
	};
	
}