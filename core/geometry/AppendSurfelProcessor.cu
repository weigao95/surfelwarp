#include "math/DualQuaternion.hpp"
#include "math/vector_ops.hpp"
#include "core/geometry/append_surfel_collision.cuh"
#include "core/geometry/AppendSurfelProcessor.h"
#include "core/geometry/KNNSearch.h"
#include "core/geometry/KNNBruteForceLiveNodes.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	//The kernel to build the candidate surfel and finite diff vertex
	__global__ void buildCandidateSurfelAndFiniteDiffVertexKernel(
		cudaTextureObject_t depth_vertex_confid_map,
		cudaTextureObject_t depth_normal_radius_map,
		cudaTextureObject_t color_time_map,
		mat34 camera2world,
		DeviceArrayView<ushort2> candidate_pixel,
		const float finitediff_step,
		//The output
		float4* finitediff_vertex,
		float4* surfel_vertex_confid,
		float4* surfel_normal_radius,
		float4* surfel_color_time
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(idx < candidate_pixel.Size())
		{
			const ushort2 pixel = candidate_pixel[idx];
			const float4 depth_vertex_confid = tex2D<float4>(depth_vertex_confid_map, pixel.x, pixel.y);
			const float4 depth_normal_radius = tex2D<float4>(depth_normal_radius_map, pixel.x, pixel.y);
			const float4 color_time = tex2D<float4>(color_time_map, pixel.x, pixel.y);
			
			//Transform the vertex into the world frame
			const float3 vertex = camera2world.rot * depth_vertex_confid + camera2world.trans;
			const float3 normal = camera2world.rot * depth_normal_radius;

			//Write to finitediff vertex buffer
			const auto offset = idx * 4;
			finitediff_vertex[offset + 0] = make_float4(vertex.x, vertex.y, vertex.z, depth_vertex_confid.w); // Note that the last element is confidence
			finitediff_vertex[offset + 1] = make_float4(vertex.x + finitediff_step, vertex.y, vertex.z, 1.0f);
			finitediff_vertex[offset + 2] = make_float4(vertex.x, vertex.y + finitediff_step, vertex.z, 1.0f);
			finitediff_vertex[offset + 3] = make_float4(vertex.x, vertex.y, vertex.z + finitediff_step, 1.0f);

			//Write to surfel array
			surfel_vertex_confid[idx] = make_float4(vertex.x, vertex.y, vertex.z, depth_vertex_confid.w);
			surfel_normal_radius[idx] = make_float4(normal.x, normal.y, normal.z, depth_normal_radius.w);
			surfel_color_time[idx] = color_time;
		}
	}


	struct SurfelCandidateFilterDevice {
		//The node coordinate of the
		struct {
			const float4* live_node_coords;
			const float4* reference_node_coords;
			const DualQuaternion* node_se3;
		} warpfield_input;

		//The finite diff data input
		struct {
			DeviceArrayView<float4> vertex_finitediff_array;
			const ushort4* vertex_finitediff_knn;
			const float4* vertex_finitediff_knnweight;
			float finitediff_step;
		} vertex_input;

		//The output indicator
		mutable unsigned* candidate_validity_indicator;
		mutable ushort4* candidate_knn;
		mutable float4* candidate_knn_weight;


		__host__ __device__ __forceinline__ float min_distance2node_squared(
			const float4& vertex,
			const ushort4& knn
		) const {
			//The first knn
			float4 node = warpfield_input.live_node_coords[knn.x];
			float min_dist_square = squared_norm_xyz(node - vertex);

			//The second knn
			node = warpfield_input.live_node_coords[knn.y];
			min_dist_square = min(min_dist_square, squared_norm_xyz(node - vertex));

			//The third knn
			node = warpfield_input.live_node_coords[knn.z];
			min_dist_square = min(min_dist_square, squared_norm_xyz(node - vertex));

			//The forth knn
			node = warpfield_input.live_node_coords[knn.w];
			min_dist_square = min(min_dist_square, squared_norm_xyz(node - vertex));

			return min_dist_square;
		}

		__host__ __device__ __forceinline__ float average_distance2node_squared(
			const float4& vertex,
			const ushort4& knn
		) const {
			//The first knn
			float4 node = warpfield_input.live_node_coords[knn.x];
			float avg_dist_square = squared_norm_xyz(node - vertex);

			//The second knn
			node = warpfield_input.live_node_coords[knn.y];
			avg_dist_square += squared_norm_xyz(node - vertex);

			//The third knn
			node = warpfield_input.live_node_coords[knn.z];
			avg_dist_square += squared_norm_xyz(node - vertex);

			//The forth knn
			node = warpfield_input.live_node_coords[knn.w];
			avg_dist_square += squared_norm_xyz(node - vertex);

			//Always count for four nodes
			return 0.25f * avg_dist_square;
		}

		__host__ __device__ __forceinline__ bool is_skinning_consistent(
			const ushort4& knn
		) const {
			float live_pairwise_distance[6];
			float canonical_pairwise_distance[6];
			const unsigned short* knn_array = (const unsigned short*)&knn;
			int shift = 0;
			for (auto i = 0; i < 4; i++) {
				for (auto j = i + 1; j < 4; j++) {
					live_pairwise_distance[shift] = squared_norm_xyz(warpfield_input.live_node_coords[knn_array[i]] - warpfield_input.live_node_coords[knn_array[j]]);
					canonical_pairwise_distance[shift] = squared_norm_xyz(warpfield_input.reference_node_coords[knn_array[i]] - warpfield_input.reference_node_coords[knn_array[j]]);
					shift++;
				}
			}

			bool consistent_skinning = true;
			for (auto i = 0; i < 6; i++) {
				if (live_pairwise_distance[i] < 0.64f * canonical_pairwise_distance[i]) consistent_skinning = false;
			}
			return consistent_skinning;
		}


		__device__ __forceinline__ void processFiltering() const {
			const auto candidate_idx = threadIdx.x + blockIdx.x * blockDim.x;
			const auto offset = candidate_idx * 4;
			if(offset >= vertex_input.vertex_finitediff_array.Size()) return;

			//Load the vertex
			const float4 vertex = vertex_input.vertex_finitediff_array[offset];
			const ushort4 vertex_knn = vertex_input.vertex_finitediff_knn[offset];
			const float4 vertex_knnweight = vertex_input.vertex_finitediff_knnweight[offset];

			//The written marker
			unsigned candidate_valid = 1;

			//Check distance
			if(min_distance2node_squared(vertex, vertex_knn) >= 4 * d_node_radius_square) candidate_valid = 0;

			//Check the consistent of skinning
			if(!is_skinning_consistent(vertex_knn)) candidate_valid = 0;

			//Check collision
			{
				//Load the data
				float4 finitediff_vertex[3], finitediff_weight[3];
				ushort4 finitediff_knn[3];
				for(auto i = 0; i < 3; i++) {
					finitediff_vertex[i] = vertex_input.vertex_finitediff_array[offset + 1 + i];
					finitediff_knn[i] = vertex_input.vertex_finitediff_knn[offset + 1 + i];
					finitediff_weight[i] = vertex_input.vertex_finitediff_knnweight[offset + 1 + i];
				}

				//Check it
				const bool compression = is_compressive_mapped(
					vertex, 
					vertex_knn, vertex_knnweight, 
					finitediff_vertex, 
					finitediff_knn, finitediff_weight, 
					warpfield_input.node_se3,
					vertex_input.finitediff_step
				);
				if(compression) candidate_valid = 0;
			}

			//Write to output
			candidate_validity_indicator[candidate_idx] = candidate_valid;
			candidate_knn[candidate_idx] = vertex_knn;
			candidate_knn_weight[candidate_idx] = vertex_knnweight;
		}
	};
	
	__global__ void filterCandidateSurfelKernel(
		const SurfelCandidateFilterDevice filter
	) {
		filter.processFiltering();
	}

} // device
} // surfelwarp


/* The method to build vertex. Using either indicator or pixels. The indicator will case sync
 */
void surfelwarp::AppendSurfelProcessor::BuildSurfelAndFiniteDiffVertex(cudaStream_t stream) {
	//The size of array contains the element itself
	const auto num_candidate = m_surfel_candidate_pixel.Size();
	m_surfel_vertex_confid.ResizeArrayOrException(num_candidate);
	m_surfel_normal_radius.ResizeArrayOrException(num_candidate);
	m_surfel_color_time.ResizeArrayOrException(num_candidate);
	m_candidate_vertex_finite_diff.ResizeArrayOrException(num_candidate * kNumFiniteDiffVertex);
	
	//The appended surfel size is zero
	if(num_candidate == 0) {
		LOG(INFO) << "There is no appended surfel";
		return;
	}
	
	//Invoke the kernel
	dim3 blk(64);
	dim3 grid(divUp(m_surfel_candidate_pixel.Size(), blk.x));
	device::buildCandidateSurfelAndFiniteDiffVertexKernel<<<grid, blk, 0, stream>>>(
		m_observation.vertex_confid_map,
		m_observation.normal_radius_map,
		m_observation.color_time_map,
		m_camera2world,
		m_surfel_candidate_pixel,
		kFiniteDiffStep,
		//The output
		m_candidate_vertex_finite_diff.Ptr(),
		m_surfel_vertex_confid.Ptr(),
		m_surfel_normal_radius.Ptr(),
		m_surfel_color_time.Ptr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
	
	//Download the data
	//std::vector<float4> h_candidate_finitediff_vec;
	//m_candidate_vertex_finite_diff.ArrayView().Download(h_candidate_finitediff_vec);
	//SURFELWARP_CHECK(h_candidate_finitediff_vec.size() == m_surfel_candidate_pixel.Size() * 4);
#endif
}

void surfelwarp::AppendSurfelProcessor::SkinningFiniteDifferenceVertex(cudaStream_t stream) {
	//Resize the array
	m_candidate_vertex_finitediff_knn.ResizeArrayOrException(m_candidate_vertex_finite_diff.ArraySize());
	m_candidate_vertex_finitediff_knnweight.ResizeArrayOrException(m_candidate_vertex_finite_diff.ArraySize());
	
	//If there is not surfel candidate
	if(m_candidate_vertex_finitediff_knn.ArraySize() == 0) {
		return;
	}
	
	m_live_node_skinner->Skinning(
		m_candidate_vertex_finite_diff.ArrayView(),
		m_candidate_vertex_finitediff_knn.ArraySlice(), m_candidate_vertex_finitediff_knnweight.ArraySlice(),
		stream
	);
	
	//Check the result of skinning: seems correct
	/*KNNSearch::CheckKNNSearch(
		m_warpfield_input.live_node_coords,
		m_candidate_vertex_finite_diff.ArrayView(),
		m_candidate_vertex_finitediff_knn.ArrayView()
	);*/
}

void surfelwarp::AppendSurfelProcessor::FilterCandidateSurfels(cudaStream_t stream) {
	//Resize the indicator
	m_candidate_surfel_validity_indicator.ResizeArrayOrException(m_surfel_candidate_pixel.Size());
	m_surfel_knn.ResizeArrayOrException(m_surfel_candidate_pixel.Size());
	m_surfel_knn_weight.ResizeArrayOrException(m_surfel_candidate_pixel.Size());
	
	//Check if the size is zero
	if(m_surfel_knn.ArraySize() == 0) return;
	
	//Construct the filter
	device::SurfelCandidateFilterDevice filter;
	
	filter.warpfield_input.live_node_coords = m_warpfield_input.live_node_coords.RawPtr();
	filter.warpfield_input.reference_node_coords = m_warpfield_input.reference_node_coords.RawPtr();
	filter.warpfield_input.node_se3 = m_warpfield_input.node_se3.RawPtr();
	
	filter.vertex_input.vertex_finitediff_array = m_candidate_vertex_finite_diff.ArrayView();
	filter.vertex_input.vertex_finitediff_knn = m_candidate_vertex_finitediff_knn.Ptr();
	filter.vertex_input.vertex_finitediff_knnweight = m_candidate_vertex_finitediff_knnweight.Ptr();
	filter.vertex_input.finitediff_step = kFiniteDiffStep;
	
	filter.candidate_validity_indicator = m_candidate_surfel_validity_indicator.Ptr();
	filter.candidate_knn = m_surfel_knn.Ptr();
	filter.candidate_knn_weight = m_surfel_knn_weight.Ptr();
	
	//Seems now ready for device code
	dim3 blk(64);
	dim3 grid(m_surfel_candidate_pixel.Size(), blk.x);
	device::filterCandidateSurfelKernel<<<grid, blk, 0, stream>>>(filter);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Do a prefix sum on the indicator
	m_candidate_surfel_validity_prefixsum.InclusiveSum(m_candidate_surfel_validity_indicator.ArrayView(), stream);
	
	//Debug
	/*KNNSearch::CheckKNNSearch(
		m_warpfield_input.live_node_coords,
		m_surfel_vertex_confid.ArrayView(),
		m_surfel_knn.ArrayView()
	);*/
	
	//LOG(INFO) << "The number of valid appended surfels is " << numNonZeroElement(m_candidate_surfel_validity_indicator.ArrayView());
}


