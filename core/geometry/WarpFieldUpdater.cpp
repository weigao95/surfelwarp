//
// Created by wei on 5/9/18.
//

#include "common/Constants.h"
#include "core/geometry/WarpFieldUpdater.h"
#include "core/geometry/KNNSearch.h"
#include <cmath>


void surfelwarp::WarpFieldUpdater::UpdateWarpFieldFromUncoveredCandidate(
	surfelwarp::WarpField &warp_field,
	const std::vector<float4> &node_candidate,
	cudaStream_t stream
) {
	//First append them into a new vector
	std::vector<float4> added_node_candidates; added_node_candidates.clear();
	for (auto i = 0; i < node_candidate.size(); i++) {
		const float4 point = make_float4(node_candidate[i].x, node_candidate[i].y, node_candidate[i].z, 1.0f);
		if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z)) {
			LOG(FATAL) << "Nan in node candidate";
			continue;
		}
		
		// Brute-force check
		bool isNode = true;
		for (auto j = 0; j < added_node_candidates.size(); j++) {
			const auto& node = added_node_candidates[j];
			if (squared_norm(node - point) <= d_node_radius_square) {
				isNode = false;
				break;
			}
		}
		
		//If this is node
		if (isNode) {
			added_node_candidates.push_back(point);
		}
	}
	
	//Update warpfield, knn and weight
	auto& h_node_se3 = warp_field.m_node_se3.HostArray();
	std::vector<ushort4> h_added_knn;
	std::vector<float4> h_added_knnweight;
	std::vector<float4> added_nodes;
	
	//Iterate over points
	for (auto i = 0; i < added_node_candidates.size(); i++) {
		DualQuaternion dq;
		ushort4 knn; float4 knn_weight;
		const bool valid = ComputeSE3AtPointHost(warp_field, added_node_candidates[i], dq, knn, knn_weight);
		if(valid) {
			added_nodes.push_back(added_node_candidates[i]);
			h_node_se3.push_back(dq);
			h_added_knn.push_back(knn);
			h_added_knnweight.push_back(knn_weight);
		}
	}
	
	//Sync it
	warp_field.m_node_se3.SynchronizeToDevice(stream);
	
	//Only copy the added part to device
	{
		const auto node_offset = warp_field.m_reference_node_coords.HostArraySize();
		ushort4* added_knn_dev = warp_field.m_node_knn.Ptr() + node_offset;
		cudaSafeCall(cudaMemcpyAsync(added_knn_dev, h_added_knn.data(), sizeof(ushort4) * h_added_knn.size(), cudaMemcpyHostToDevice, stream));
		float4* added_knnweight_dev = warp_field.m_node_knn_weight.Ptr() + node_offset;
		cudaSafeCall(cudaMemcpyAsync(added_knnweight_dev, h_added_knnweight.data(), sizeof(float4) * h_added_knn.size(), cudaMemcpyHostToDevice, stream));
		warp_field.m_node_knn.ResizeArrayOrException(warp_field.m_node_se3.HostArraySize());
		warp_field.m_node_knn_weight.ResizeArrayOrException(warp_field.m_node_se3.HostArraySize());
	}
	
	//Update the refernce nodes
	auto& h_ref_nodes = warp_field.m_reference_node_coords.HostArray();
	for(auto i = 0; i < added_nodes.size(); i++) {
		const auto& node = added_nodes[i];
		h_ref_nodes.push_back(node);
	}
	warp_field.m_reference_node_coords.SynchronizeToDevice(stream);
	
	//Resize the device array
	warp_field.m_live_node_coords.ResizeArrayOrException(warp_field.m_node_se3.HostArraySize());
	warp_field.CheckAndGetNodeSize();
	
	//Debug method
	//cudaSafeCall(cudaStreamSynchronize(stream));
	//CheckUpdatedWarpField(warp_field, h_added_knn, h_added_knnweight);
	//LOG(INFO) << "The number of appended node is " << h_added_knn.size();
}


void surfelwarp::WarpFieldUpdater::CheckUpdatedWarpField(
	const WarpField& warp_field,
	const std::vector<ushort4>& h_added_knn,
	const std::vector<float4>& h_added_knnweight
) {
	LOG(INFO) << "The number of appended nodes is " << h_added_knn.size();
	
	//Check the knn: correct here
	const float4* skinning_from_vertex = warp_field.m_reference_node_coords.DevicePtr();
	const auto skinning_from_size = warp_field.m_reference_node_coords.DeviceArraySize() - h_added_knn.size();
	DeviceArrayView<float4> skinning_from = DeviceArrayView<float4>(skinning_from_vertex, skinning_from_size);
	KNNSearch::CheckKNNSearch(
		skinning_from,
		warp_field.m_reference_node_coords.DeviceArrayReadOnly(), warp_field.m_node_knn.ArrayView()
	);
	
	//Download the knn for check
	std::vector<ushort4> h_knn;
	std::vector<float4> h_knnweight;
	warp_field.m_node_knn.ArrayReadOnly().Download(h_knn);
	warp_field.m_node_knn_weight.ArrayReadOnly().Download(h_knnweight);
	unsigned newnode_offset = h_knn.size() - h_added_knn.size();
	const ushort4 newnode_knn = h_knn[newnode_offset];
	const float4 newnode_knnweight = h_knnweight[newnode_offset];
	LOG(INFO) << "The knn at newnode offset " << newnode_offset << " is " << newnode_knn.x << " " << newnode_knn.y << " "<< newnode_knn.z << " "<< newnode_knn.w << " ";
	LOG(INFO) << "The knn weight at newnode offset " << newnode_offset << " is " << newnode_knnweight.x << " " << newnode_knnweight.y << " "<< newnode_knnweight.z << " "<< newnode_knnweight.w << " ";
	
	
	//The added knn
	LOG(INFO) << "The append knn is " << h_added_knn[0].x << " " << h_added_knn[0].y << " "<< h_added_knn[0].z << " "<< h_added_knn[0].w << " ";
	LOG(INFO) << "The append knn weight is " << h_added_knnweight[0].x << " " << h_added_knnweight[0].y << " "<< h_added_knnweight[0].z << " "<< h_added_knnweight[0].w << " ";
}

bool surfelwarp::WarpFieldUpdater::ComputeSE3AtPointHost(
	const WarpField &warp_field,
	const float4 &point,
	DualQuaternion &dq,
	ushort4 &knn, float4 &knn_weight
) {
	// Collect knn at host
	std::vector<int> node_idx_array(warp_field.m_reference_node_coords.HostArray().size());
	const auto& coord_array = warp_field.m_reference_node_coords.HostArray();
	int count = 0;
	std::generate_n(node_idx_array.begin(), node_idx_array.size(), [&count]()->int { return count++; });
	std::partial_sort(node_idx_array.begin(), node_idx_array.begin() + 4, node_idx_array.end(),
	                  [&coord_array, &point](const int& l, const int& r) -> bool {
		                  return squared_norm_xyz(coord_array[l] - point) < squared_norm_xyz(coord_array[r] - point);
	                  });
	
	//The nearest neighbour is ready, now average on the dual quaternion
	DualQuaternion dq_average;
	float weight_sum = 0.0f;
	unsigned short knn_array[4]; float knn_weight_array[4];
	dq_average.set_zero();
	for (int nn_iter = 0; nn_iter < 4; nn_iter++) {
		const auto neighbour = node_idx_array[nn_iter];
		const auto& neighbour_node = coord_array[neighbour];
		const float distance_square = squared_norm_xyz(neighbour_node - point);
		const float weight = expf(-0.5f * distance_square / d_node_radius_square);
		const DualQuaternion node_dq = warp_field.m_node_se3.HostArray()[neighbour];
		dq_average += DualNumber(weight, 0) * node_dq;
		knn_array[nn_iter] = neighbour;
		knn_weight_array[nn_iter] = weight;
		weight_sum += weight;
	}
	
	//Do a normalization on weight
	for(auto i = 0; i < 4; i++) {
		if(weight_sum < 1e-1f) {
			LOG(WARNING) << "Invalid wight in node KNN";
			knn_weight_array[i] = 0.0f;
			return false;
		}
		else knn_weight_array[i] *= (1.0f / weight_sum);
	}
	
	dq = dq_average.normalized();

	//Note to use inverse ordering
	knn = make_ushort4(knn_array[3], knn_array[2], knn_array[1], knn_array[0]);
	knn_weight = make_float4(knn_weight_array[3], knn_weight_array[2], knn_weight_array[1], knn_weight_array[0]);
	return true;
}

void surfelwarp::WarpFieldUpdater::InitializeReferenceNodesAndSE3FromCandidates(
	WarpField &warp_field,
	const std::vector<float4> &node_candidate,
	cudaStream_t stream
) {
	//Do no touch other elements
	warp_field.m_reference_node_coords.ClearArray();
	warp_field.m_node_se3.ClearArray();
	
	//Collect elements
	const float sample_distance_square = (Constants::kNodeSamplingRadius) * (Constants::kNodeSamplingRadius);
	const auto& h_candidate = node_candidate;
	auto& h_nodes = warp_field.m_reference_node_coords.HostArray();
	auto& h_node_se3 = warp_field.m_node_se3.HostArray();
	
	//The host iterations
	for(auto vert_iter = 0; vert_iter < h_candidate.size(); vert_iter++) {
		const float4& candidate_point = h_candidate[vert_iter];
		const float4 point = make_float4(candidate_point.x, candidate_point.y, candidate_point.z, 1.0f);
		bool is_node = true;
		for(auto node_iter = 0; node_iter < h_nodes.size(); node_iter++) {
			if(squared_norm_xyz(point - h_nodes[node_iter]) < sample_distance_square) {
				is_node = false;
				break;
			}
		}
		
		//Update the node position and se3
		if(is_node) {
			h_nodes.emplace_back(make_float4(point.x, point.y, point.z, 1.0f));
			h_node_se3.emplace_back(DualQuaternion(Quaternion(1, 0, 0, 0), Quaternion(0, 0, 0, 0)));
		}
	}
	
	//Sync to device
	warp_field.m_reference_node_coords.SynchronizeToDevice(stream);
	warp_field.m_node_se3.SynchronizeToDevice(stream);
	
	//Resize other arrays
	const auto num_nodes = warp_field.m_reference_node_coords.HostArraySize();
	warp_field.ResizeDeviceArrayToNodeSize(num_nodes);
}


