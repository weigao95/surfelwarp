#include "core/warp_solver/SparseCorrespondenceHandler.h"
#include "SparseCorrespondenceHandler.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	enum {
		window_halfsize = 1,
	};

	__device__ ushort2 validGeometryPixelInWindow(
		cudaTextureObject_t index_map,
		unsigned short center_x, unsigned short center_y
	) {
		ushort2 valid_pixel = make_ushort2(0xFFFF, 0xFFFF);

		//Perform a window search
		for(auto y = center_y - window_halfsize; y <= center_y + window_halfsize; y++) {
			for(auto x = center_x - window_halfsize; x <= center_x + window_halfsize; x++) {
				if(tex2D<unsigned>(index_map, x, y) != 0xFFFFFFFF) {
					valid_pixel.x = x;
					valid_pixel.y = y;
					break;
				}
			}
		}

		//Always prefer the center one
		if(tex2D<unsigned>(index_map, center_x, center_y) != 0xFFFFFFFF) {
			valid_pixel.x = center_x;
			valid_pixel.y = center_y;
		}

		//Return it
		return valid_pixel;
	}

	__device__ ushort2 validDepthPixelInWindow(
		cudaTextureObject_t depth_vertex_map,
		unsigned short center_x, unsigned short center_y
	) {
		ushort2 valid_pixel = make_ushort2(0xFFFF, 0xFFFF);

		//Perform a window search
		for(auto y = center_y - window_halfsize; y <= center_y + window_halfsize; y++) {
			for(auto x = center_x - window_halfsize; x <= center_x + window_halfsize; x++) {
				const float4 vertex = tex2D<float4>(depth_vertex_map, x, y);
				if(!is_zero_vertex(vertex)) {
					valid_pixel.x = x;
					valid_pixel.y = y;
					break;
				}
			}
		}

		//Always prefer the center one
		const float4 center_vertex = tex2D<float4>(depth_vertex_map, center_x, center_y);
		if(!is_zero_vertex(center_vertex)) {
			valid_pixel.x = center_x;
			valid_pixel.y = center_y;
		}

		//Return it
		return valid_pixel;
	}

	__global__ void chooseValidPixelKernel(
		DeviceArrayView<ushort4> candidate_pixel_pairs,
		cudaTextureObject_t depth_vertex_map,
		cudaTextureObject_t index_map,
		unsigned rows, unsigned cols,
		unsigned* valid_indicator,
		ushort4* valid_pixel_pairs
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < candidate_pixel_pairs.Size())
		{
			const auto candidate_pair = candidate_pixel_pairs[idx];
			const auto geometry_pixel = validGeometryPixelInWindow(index_map, candidate_pair.x, candidate_pair.y);
			const auto depth_pixel = validDepthPixelInWindow(depth_vertex_map, candidate_pair.z, candidate_pair.w);
			if(geometry_pixel.x < cols && geometry_pixel.y < rows && depth_pixel.x < cols && depth_pixel.y < rows) {
				valid_indicator[idx] = 1;
				valid_pixel_pairs[idx] = make_ushort4(geometry_pixel.x, geometry_pixel.y, depth_pixel.x, depth_pixel.y);
			}
			else {
				valid_indicator[idx] = 0;
			}
		}
	}
	
	__global__ void compactQueryValidPairsKernel(
		cudaTextureObject_t depth_vertex_map,
		cudaTextureObject_t reference_vertex_map,
		const DeviceArrayView2D<KNNAndWeight> knn_map,
		const DeviceArrayView<unsigned> valid_indicator,
		const unsigned* prefixsum_indicator,
		const ushort4* valid_pixel_pairs,
		const mat34 camera2world,
		float4* target_vertex_array,
		float4* reference_vertex_array,
		ushort4* knn_array,
		float4* knn_weight_array
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx >= valid_indicator.Size()) return;

		if(valid_indicator[idx] != 0) {
			const auto offset = prefixsum_indicator[idx] - 1;
			const auto pixel_pair = valid_pixel_pairs[idx];
			const float4 reference_vertex = tex2D<float4>(reference_vertex_map, pixel_pair.x, pixel_pair.y);
			const float4 depth_vertex = tex2D<float4>(depth_vertex_map, pixel_pair.z, pixel_pair.w);
			const auto knn = knn_map(pixel_pair.y, pixel_pair.x); //KNN must be valid
			
			//Compute the target vertex
			const float3 depth_v3 = make_float3(depth_vertex.x, depth_vertex.y, depth_vertex.z);
			const float3 target_v3 = camera2world.rot * depth_v3 + camera2world.trans;

			//Write to output
			target_vertex_array[offset] = make_float4(target_v3.x, target_v3.y, target_v3.z, 1.0f);
			reference_vertex_array[offset] = reference_vertex;
			knn_array[offset] = knn.knn;
			knn_weight_array[offset] = knn.weight;
		}
	}


	//Forward warp the vertex for better computation of jacobian
	__global__ void forwardWarpFeatureVertexKernel(
		DeviceArrayView<float4> reference_vertex_array,
		const ushort4* vertex_knn_array,
		const float4* vertex_knnweight_array,
		const DualQuaternion* node_se3,
		float4* warped_vertex_array
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(idx < reference_vertex_array.Size()) {
			const float4 reference_vertex = reference_vertex_array[idx];
			const ushort4 knn = vertex_knn_array[idx];
			const float4 knnweight = vertex_knnweight_array[idx];
			DualQuaternion dq = averageDualQuaternion(node_se3, knn, knnweight);
			const mat34 se3 = dq.se3_matrix();
			const float3 warped_vertex = se3.rot * reference_vertex + se3.trans;
			warped_vertex_array[idx] = make_float4(warped_vertex.x, warped_vertex.y, warped_vertex.z, 1.0f);
		}
	}

} // namespace device
} // namespace surfelwarp



void surfelwarp::SparseCorrespondenceHandler::ChooseValidPixelPairs(cudaStream_t stream) {
	m_valid_pixel_indicator.ResizeArrayOrException(m_observations.correspond_pixel_pairs.Size());
	m_corrected_pixel_pairs.ResizeArrayOrException(m_observations.correspond_pixel_pairs.Size());
	
	//The correspondence array might be empty
	if(m_valid_pixel_indicator.ArraySize() == 0) return;
	
	dim3 blk(64);
	dim3 grid(divUp(m_valid_pixel_indicator.ArraySize(), blk.x));
	const auto rows = m_geometry_maps.knn_map.Rows();
	const auto cols = m_geometry_maps.knn_map.Cols();
	device::chooseValidPixelKernel<<<grid, blk, 0, stream>>>(
		m_observations.correspond_pixel_pairs,
		m_observations.depth_vertex_map,
		m_geometry_maps.index_map,
		rows, cols,
		m_valid_pixel_indicator,
		m_corrected_pixel_pairs
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::SparseCorrespondenceHandler::CompactQueryPixelPairs(cudaStream_t stream) {
	//The correspondence array might be empty
	if(m_valid_pixel_indicator.ArraySize() == 0) return;
	
	//Inclusive sum
	m_valid_pixel_prefixsum.InclusiveSum(m_valid_pixel_indicator.ArrayView(), stream);

#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Choose it
	dim3 blk(64);
	dim3 grid(divUp(m_valid_pixel_indicator.ArraySize(), blk.x));
	device::compactQueryValidPairsKernel<<<grid, blk, 0, stream>>>(
		m_observations.depth_vertex_map,
		m_geometry_maps.reference_vertex_map,
		m_geometry_maps.knn_map,
		//Prefix-sum information
		m_valid_pixel_indicator.ArrayView(),
		m_valid_pixel_prefixsum.valid_prefixsum_array.ptr(),
		m_corrected_pixel_pairs.Ptr(),
		m_camera2world,
		//The output
		m_valid_target_vertex.Ptr(),
		m_valid_reference_vertex.Ptr(),
		m_valid_vertex_knn.Ptr(),
		m_valid_knn_weight.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::SparseCorrespondenceHandler::QueryCompactedArraySize(cudaStream_t stream) {
	//The correspondence array might be empty
	if(m_valid_pixel_indicator.ArraySize() == 0) {
		m_valid_target_vertex.ResizeArrayOrException(0);
		m_valid_reference_vertex.ResizeArrayOrException(0);
		m_valid_vertex_knn.ResizeArrayOrException(0);
		m_valid_knn_weight.ResizeArrayOrException(0);
		return;
	}
	
	//Non-empty array
	//unsigned valid_array_size;
	cudaSafeCall(cudaMemcpyAsync(
		m_correspondence_array_size,
		m_valid_pixel_prefixsum.valid_prefixsum_array.ptr() + m_valid_pixel_prefixsum.valid_prefixsum_array.size() - 1,
		sizeof(unsigned),
		cudaMemcpyDeviceToHost,
		stream
	));
	
	//Sync before use
	cudaSafeCall(cudaStreamSynchronize(stream));
	
	//LOG(INFO) << "The number of valid pixel pairs is " << valid_array_size;
	
	//Correct the size
	m_valid_target_vertex.ResizeArrayOrException(*m_correspondence_array_size);
	m_valid_reference_vertex.ResizeArrayOrException(*m_correspondence_array_size);
	m_valid_vertex_knn.ResizeArrayOrException(*m_correspondence_array_size);
	m_valid_knn_weight.ResizeArrayOrException(*m_correspondence_array_size);
}


/* The method to build the term 2 jacobian map
 */
void surfelwarp::SparseCorrespondenceHandler::forwardWarpFeatureVertex(cudaStream_t stream) {
	//Correct the size
	m_valid_warped_vertex.ResizeArrayOrException(m_valid_reference_vertex.ArraySize());

	//Do a forward warp
	dim3 blk(128);
	dim3 grid(divUp(m_valid_reference_vertex.ArraySize(), blk.x));
	device::forwardWarpFeatureVertexKernel<<<grid, blk, 0, stream>>>(
		m_valid_reference_vertex.ArrayView(), 
		m_valid_vertex_knn.Ptr(), m_valid_knn_weight.Ptr(),
		m_node_se3.RawPtr(), 
		m_valid_warped_vertex.Ptr()
	);


	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


void surfelwarp::SparseCorrespondenceHandler::BuildTerm2Jacobian(cudaStream_t stream) {
	forwardWarpFeatureVertex(stream);
}


surfelwarp::Point2PointICPTerm2Jacobian surfelwarp::SparseCorrespondenceHandler::Term2JacobianMap() const {
	Point2PointICPTerm2Jacobian term2jacobian;
	term2jacobian.target_vertex = m_valid_target_vertex.ArrayView();
	term2jacobian.reference_vertex = m_valid_reference_vertex.ArrayView();
	term2jacobian.knn = m_valid_vertex_knn.ArrayView();
	term2jacobian.knn_weight = m_valid_knn_weight.ArrayView();
	term2jacobian.node_se3 = m_node_se3;
	term2jacobian.warped_vertex = m_valid_warped_vertex.ArrayView();
	
	//Check the size
	SURFELWARP_CHECK_EQ(term2jacobian.target_vertex.Size(), term2jacobian.reference_vertex.Size());
	SURFELWARP_CHECK_EQ(term2jacobian.target_vertex.Size(), term2jacobian.knn.Size());
	SURFELWARP_CHECK_EQ(term2jacobian.target_vertex.Size(), term2jacobian.knn_weight.Size());
	SURFELWARP_CHECK_EQ(term2jacobian.target_vertex.Size(), term2jacobian.warped_vertex.Size());

	//Return it
	return term2jacobian;
}




