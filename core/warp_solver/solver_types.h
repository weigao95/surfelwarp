#pragma once

#include "common/logging.h"
#include "common/ArrayView.h"
#include "common/common_utils.h"
#include "math/DualQuaternion.hpp"

#include <ostream>

namespace surfelwarp {


	/**
	 * \brief The gradient of some scalar cost 
	 *        towards the twist parameters of node 
	 *        SE(3) INCREMENT, while the node SE(3)
	 *        itself is parameterized by DualQuaternion
	 */
	struct TwistGradientOfScalarCost {
		float3 rotation;
		float3 translation;
		
		//Constant times operator
		__host__ TwistGradientOfScalarCost operator*(const float& value) const {
			TwistGradientOfScalarCost timed_twist = *this;
			timed_twist.rotation *= value;
			timed_twist.translation *= value;
			return timed_twist;
		}

		//Dot with a size 6 array
		__host__ __device__ float dot(const float x[6]) const {
			const float rot_dot = x[0] * rotation.x + x[1] * rotation.y + x[2] * rotation.z;
			const float trans_dot = x[3] * translation.x + x[4] * translation.y + x[5] * translation.z;
			return rot_dot + trans_dot;
		}

		//Dot with a texture memory
		__device__ __forceinline__ float DotLinearTexture(cudaTextureObject_t x, const unsigned x_offset) const {
			const float* jacobian = (const float*)(this);
			float dot_value = 0.0f;
			for(auto i = 0; i < 6; i++) {
				dot_value += jacobian[i] * fetch1DLinear<float>(x, x_offset + i);
			}
			return dot_value;
		}
		__device__ __forceinline__ float DotArrayTexture(cudaTextureObject_t x, const unsigned x_offset) const {
			const float* jacobian = (const float*)(this);
			float dot_value = 0.0f;
			for(auto i = 0; i < 6; i++) {
				dot_value += jacobian[i] * fetch1DArray<float>(x, x_offset + i);
			}
			return dot_value;
		}
	};
	
	
	/* The gradient with multiple scalar cost
	 */
	template<int NumChannels = 3>
	struct TwistGradientChannel {
		TwistGradientOfScalarCost twist_gradient[NumChannels];
	};
	
	/**
	 * \brief The Term2Jacobian structs, as its name suggested, 
	 *        provide enough information to compute the gradient
	 *        of the cost from a given term index w.r.t all the
	 *        nodes that this term is involved. 
	 *        Note that each term may have ONE or MORE scalar
	 *        costs. In case of multiple scalar costs, the jacobian
	 *        CAN NOT be combined.
	 *        The term2jacobian should implemented on device, but
	 *        may provide host implementation for debug checking.
	 */
	struct ScalarCostTerm2Jacobian {
		DeviceArrayView<ushort4> knn_array;
		DeviceArrayView<float4> knn_weight_array;
		DeviceArrayView<float> residual_array;
		DeviceArrayView<TwistGradientOfScalarCost> twist_gradient_array;
		
		//Simple sanity check
		__host__ __forceinline__ void check_size() const {
			SURFELWARP_CHECK_EQ(knn_array.Size(), knn_weight_array.Size());
			SURFELWARP_CHECK_EQ(knn_array.Size(), residual_array.Size());
			SURFELWARP_CHECK_EQ(knn_array.Size(), twist_gradient_array.Size());
		}
		
		__host__ void computeFlattenJacobian(std::vector<float>& jacobian) const {
			jacobian.resize(twist_gradient_array.Size() * 6 * 4);
			
			//Download the data
			std::vector<float4> h_knn_weight;
			knn_weight_array.Download(h_knn_weight);
			std::vector<TwistGradientOfScalarCost> h_twist_gradient;
			twist_gradient_array.Download(h_twist_gradient);
			
			//Do iteration
			for(auto i = 0; i < twist_gradient_array.Size(); i++) {
				TwistGradientOfScalarCost scalar_gradient = h_twist_gradient[i];
				float4 scalar_knn_weight = h_knn_weight[i];
				const float* scalar_gradient_flatten = (const float*)&scalar_gradient;
				const float* scalar_knn_weight_flatten = (const float*)&scalar_knn_weight;
				for(auto nn_iter = 0; nn_iter < 4; nn_iter++) {
					for(auto k = 0; k < 6; k++) {
						const auto offset = 24 * i + 6 * nn_iter + k;
						jacobian[offset] = scalar_gradient_flatten[k] * scalar_knn_weight_flatten[nn_iter];
					}
				}
			}
		}
	};
	
	//These are all scalar cost term types
	using DenseDepthTerm2Jacobian = ScalarCostTerm2Jacobian;
	using DensityMapTerm2Jacobian = ScalarCostTerm2Jacobian;
	using ForegroundMaskTerm2Jacobian = ScalarCostTerm2Jacobian;
	
	
	/**
	 * \brief It seems cheaper to compute the jacobian online
	 *        for smooth term.
	 */
	struct NodeGraphSmoothTerm2Jacobian {
		DeviceArrayView<ushort2> node_graph;
		DeviceArrayView<float3> Ti_xj;
		DeviceArrayView<float3> Tj_xj;
		DeviceArrayView<unsigned char> validity_indicator;

		//These are deprecated and should not be used
		DeviceArrayView<float4> reference_node_coords;
		DeviceArrayView<DualQuaternion> node_se3;
	};
	
	
	struct Point2PointICPTerm2Jacobian {
		DeviceArrayView<float4> target_vertex; //Query from depth vertex map
		DeviceArrayView<float4> warped_vertex;
		DeviceArrayView<ushort4> knn;
		DeviceArrayView<float4> knn_weight;
		
		
		//These are deprecated and should not be used
		DeviceArrayView<float4> reference_vertex; //Query from rendered vertex map
		DeviceArrayView<DualQuaternion> node_se3;
	};
	
	
	/* The collective term2jacobian maps
	 */
	struct Term2JacobianMaps {
		DenseDepthTerm2Jacobian dense_depth_term;
		NodeGraphSmoothTerm2Jacobian smooth_term;
		DensityMapTerm2Jacobian density_map_term;
		ForegroundMaskTerm2Jacobian foreground_mask_term;
		Point2PointICPTerm2Jacobian sparse_feature_term;
	};
	
	
	/* The node-wise error and weight, compute only use dense depth information
	 */
	struct NodeAlignmentError {
		DeviceArrayView<float> node_accumlated_error;
		const float* node_accumlate_weight;
		
		//The statistic method
		__host__ void errorStatistics(std::ostream& output = std::cout) const;
	};
}