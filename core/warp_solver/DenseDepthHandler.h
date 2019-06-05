//
// Created by wei on 3/29/18.
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
	
	
	class DenseDepthHandler {
	private:
		//The info from config
		int m_image_height;
		int m_image_width;
		Intrinsic m_project_intrinsic;
		
		//The info from solver
		DeviceArrayView<DualQuaternion> m_node_se3;
		DeviceArrayView2D<KNNAndWeight> m_knn_map;
		mat34 m_world2camera;
		mat34 m_camera2world;

		//The info from depth input
		struct {
			cudaTextureObject_t vertex_map;
			cudaTextureObject_t normal_map;
		} m_depth_observation;
		
		//The info from rendered maps
		struct {
			cudaTextureObject_t reference_vertex_map;
			cudaTextureObject_t reference_normal_map;
			cudaTextureObject_t index_map;
		} m_geometry_maps;
		
		//The info from image term fetcher
		ImageTermKNNFetcher::ImageTermPixelAndKNN m_potential_pixels_knn;
		
	public:
		using Ptr = std::shared_ptr<DenseDepthHandler>;
		DenseDepthHandler();
		~DenseDepthHandler() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(DenseDepthHandler);
		
		//Explicit allocate
		void AllocateBuffer();
		void ReleaseBuffer();
		
		//Set input
		void SetInputs(
			const DeviceArrayView<DualQuaternion>& node_se3,
			const DeviceArrayView2D<KNNAndWeight>& knn_map,
			cudaTextureObject_t depth_vertex_map, cudaTextureObject_t depth_normal_map,
			//The rendered maps
			cudaTextureObject_t reference_vertex_map,
			cudaTextureObject_t reference_normal_map,
			cudaTextureObject_t index_map,
			const mat34& world2camera,
			//The potential pixels,
			const ImageTermKNNFetcher::ImageTermPixelAndKNN& pixels_knn
		);
		
		//Update the se3
		void UpdateNodeSE3(DeviceArrayView<DualQuaternion> node_se3);
		
		//The processing interface for free-index solver
		void FindCorrespondenceSynced(cudaStream_t stream = 0);
	private:
		//These two should be 2D maps, flatten as the compaction is required
		DeviceArray<ushort2> m_pixel_pair_maps; //The matched depth pixel pairs, used in index free version
		DeviceArray<unsigned> m_pixel_match_indicator; //The indicator for surfel pixel, used in both version
	private:
		//The method will project the model surfel into depth image, check the
		//correspondence between them and mark corresponded surfel pixels
		void MarkMatchedPixelPairs(cudaStream_t stream = 0);
		
		/* Compact the pixel pairs
		 */
	private:
		PrefixSum m_indicator_prefixsum;
		DeviceBufferArray<ushort4> m_valid_pixel_pairs;
		DeviceBufferArray<ushort4> m_dense_depth_knn;
		DeviceBufferArray<float4> m_dense_depth_knn_weight;
	private:
		void CompactMatchedPixelPairs(cudaStream_t stream = 0);
		void compactedPairSanityCheck(DeviceArrayView<ushort4> surfel_knn_array);
	public:
		void SyncQueryCompactedArraySize(cudaStream_t stream = 0);
		
		
		/* Compute the twist jacobian
		 */
	private:
		DeviceBufferArray<float> m_term_residual;
		DeviceBufferArray<TwistGradientOfScalarCost> m_term_twist_gradient;
	public:
		void ComputeJacobianTermsFreeIndex(cudaStream_t stream = 0);
		void ComputeJacobianTermsFixedIndex(cudaStream_t stream = 0);
		DenseDepthTerm2Jacobian Term2JacobianMap() const;
		
		
		
		/* Compute the residual map and gather them into nodes. Different from previous residual
		 * The map will return non-zero value at valid pixels that doesnt have corresponded depth pixel
		 * The method is used in Reinit pipeline and visualization.
		 */
	private:
		CudaTextureSurface m_alignment_error_map;
	public:
		void ComputeAlignmentErrorMapDirect(
			const DeviceArrayView<DualQuaternion>& node_se3, const mat34& world2camera, 
			cudaTextureObject_t filter_foreground_mask, cudaStream_t stream = 0
		);
		cudaTextureObject_t GetAlignmentErrorMap() const { return m_alignment_error_map.texture; }
		
		
		/* Compute the error and accmulate them on nodes. May distribute them again on
		 * map for further use or visualization
		 */
	private:
		DeviceBufferArray<float> m_node_accumlate_error;
		DeviceBufferArray<float> m_node_accumlate_weight;
		
		//Distribute the node error on maps
		void distributeNodeErrorOnMap(cudaStream_t stream = 0);
	public:
		void ComputeNodewiseError(
			const DeviceArrayView<DualQuaternion>& node_se3,
			const mat34& world2camera,
			cudaTextureObject_t filter_foreground_mask,
			cudaStream_t stream = 0
		);
		void ComputeAlignmentErrorMapFromNode(
			const DeviceArrayView<DualQuaternion>& node_se3, const mat34& world2camera,
			cudaTextureObject_t filter_foreground_mask, cudaStream_t stream = 0
		);
		
		
		/* Accessing interface
		 */
	public:
		//The nodewise error
		NodeAlignmentError GetNodeAlignmentError() const {
			NodeAlignmentError error;
			error.node_accumlated_error = m_node_accumlate_error.ArrayView();
			error.node_accumlate_weight = m_node_accumlate_weight.Ptr();
			return error;
		}
	};
}