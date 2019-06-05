//
// Created by wei on 4/4/18.
//

#include "common/Stream.h"
#include "common/Serializer.h"
#include "common/BinaryFileStream.h"
#include "core/warp_solver/GradientCheck.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include <string>
#include <random>


//This method is actually deprecated, thus move it to here for immediate use
namespace surfelwarp { namespace device {
	
	__host__ __forceinline__ void computePointToPointICPTermJacobianResidual(
		const float4& depth_vertex_confid,
		const float4& can_vertex4,
		const ushort4& knn, const float4& knn_weight,
		//The warp field
		const DualQuaternion* device_warp_field,
		const mat34& camera2world,
		//The output
		float* residual, //[3]
		TwistGradientOfScalarCost* twist_gradient //[3]
	) {
		//Correct the size
		const float3 can_vertex = make_float3(can_vertex4.x, can_vertex4.y, can_vertex4.z);
		const float3 depth_vertex = make_float3(depth_vertex_confid.x, depth_vertex_confid.y, depth_vertex_confid.z);

		//Warp it
		DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
		const mat34 se3 = dq_average.se3_matrix();
		const float3 warped_vertex = se3.rot * can_vertex + se3.trans;

		//Warp the depth vertex to world frame
		const float3 depth_world_vertex = camera2world.rot * depth_vertex + camera2world.trans;

		//Compute the residual
		residual[0] = warped_vertex.x - depth_world_vertex.x;
		residual[1] = warped_vertex.y - depth_world_vertex.y;
		residual[2] = warped_vertex.z - depth_world_vertex.z;

		//Compute the gradient
		twist_gradient[0].translation = make_float3(1.0f, 0.0f, 0.0f);
		twist_gradient[1].translation = make_float3(0.0f, 1.0f, 0.0f);
		twist_gradient[2].translation = make_float3(0.0f, 0.0f, 1.0f);
		twist_gradient[0].rotation = make_float3(0.0f, warped_vertex.z, -warped_vertex.y);
		twist_gradient[1].rotation = make_float3(-warped_vertex.z, 0.0f, warped_vertex.x);
		twist_gradient[2].rotation = make_float3(warped_vertex.y, -warped_vertex.x, 0.0f);
	}


} // device
} // surfelwarp

void surfelwarp::GradientCheck::LoadDataFromFile(const std::string &path) {
	//Construct the input
	BinaryFileStream input_fstream(path.c_str(), BinaryFileStream::Mode::ReadOnly);
	
	//Load it
	input_fstream.SerializeRead(&m_reference_vertex_confid);
	input_fstream.SerializeRead(&m_reference_normal_radius);
	input_fstream.SerializeRead(&m_vertex_knn);
	input_fstream.SerializeRead(&m_vertex_knn_weight);
	input_fstream.SerializeRead(&m_node_graph);
	input_fstream.SerializeRead(&m_node_coordinates);

	//Correct the size
	m_init_node_se3.resize(m_node_coordinates.size());
	for(auto i = 0; i < m_init_node_se3.size(); i++) {
		m_init_node_se3[i].set_identity();
	}
	
	//Init the warp field
	m_camera2world.set_identity();
}


void surfelwarp::GradientCheck::randomInitWarpField(float max_rot, float max_trans)
{
	std::default_random_engine generator((unsigned int) time(NULL));
	std::uniform_real_distribution<float> rot_distribution(-max_rot, max_rot);
	std::uniform_real_distribution<float> trans_distribution(-max_trans, max_trans);
	for(auto i = 0; i < m_init_node_se3.size(); i++) {
		float3 twist_rot, twist_trans;
		twist_rot.x = rot_distribution(generator);
		twist_rot.y = rot_distribution(generator);
		twist_rot.z = rot_distribution(generator);
		twist_trans.x = trans_distribution(generator);
		twist_trans.y = trans_distribution(generator);
		twist_trans.z = trans_distribution(generator);
		apply_twist(twist_rot, twist_trans, m_init_node_se3[i]);
	}
	
	//Also random init the world2camera
	float3 twist_rot, twist_trans;
	twist_rot.x = rot_distribution(generator);
	twist_rot.y = rot_distribution(generator);
	twist_rot.z = rot_distribution(generator);
	twist_trans.x = trans_distribution(generator);
	twist_trans.y = trans_distribution(generator);
	twist_trans.z = trans_distribution(generator);
	DualQuaternion dq; dq.set_identity();
	apply_twist(twist_rot, twist_trans, dq);
	m_camera2world = dq.se3_matrix();
}

void surfelwarp::GradientCheck::checkSmoothTermJacobian()
{
	for(auto i = 0; i < m_node_graph.size(); i++)
	{
		const auto node_i = m_node_graph[i].x;
		const auto node_j = m_node_graph[i].y;
		const float4 xi = m_node_coordinates[node_i];
		const float4 xj = m_node_coordinates[node_j];
		DualQuaternion dq_i = m_init_node_se3[node_i];
		DualQuaternion dq_j = m_init_node_se3[node_j];
		mat34 Ti = dq_i.se3_matrix();
		mat34 Tj = dq_j.se3_matrix();

		//Compute analytical gradient
		TwistGradientOfScalarCost gradient_i[3], gradient_j[3];
		float residual[3];
		device::computeSmoothTermJacobianResidual(xi, xj, Ti, Tj, residual, gradient_i, gradient_j);

		//Compute the numerical gradient
		{
			TwistGradientOfScalarCost placeholder_i[3], placeholder_j[3];
			float new_residual[3];
			DualQuaternion dq_i_modified = dq_i;
			const float step = 0.01;
			float3 twist_rot = make_float3(step, 0, 0);
			float3 twist_trans = make_float3(0, 0, 0);
			apply_twist(twist_rot, twist_trans, dq_i_modified);
			device::computeSmoothTermJacobianResidual(xi, xj, dq_i_modified.se3_matrix(), Tj, new_residual, placeholder_i, placeholder_j);
			double d_residual_1_d_rot_0 = (new_residual[1] - residual[1]) / step;
			double d_residual_2_d_rot_0 = (new_residual[2] - residual[2]) / step;
			double diff_1 = d_residual_1_d_rot_0 - gradient_i[1].rotation.x;
			double diff_2 = d_residual_2_d_rot_0 - gradient_i[2].rotation.x;
			auto relative_diff_1 = std::abs(diff_1 / d_residual_1_d_rot_0);
			auto relative_diff_2 = std::abs(diff_2 / d_residual_2_d_rot_0);
			auto relative_diff = std::max(relative_diff_1, relative_diff_2);
			if(relative_diff_2 > 0.01f && std::abs(d_residual_2_d_rot_0) > 1e-2) {
				LOG(INFO) << "Relative difference value " << relative_diff_2 << " where the original value is " << d_residual_1_d_rot_0;
			}
		}
	}
}

void surfelwarp::GradientCheck::checkPoint2PlaneICPJacobian() {
	//The step
	const float step = 0.01;
	float3 twist_rot = make_float3(step, 0, 0);
	float3 twist_trans = make_float3(0, 0, 0);
	
	//For checking, assuming the corresponded depth vertex and normal is the next geometry vertex/normal
	const auto offset = 10;
	for(auto i = 0; i < m_reference_vertex_confid.size() - offset; i++) {
		//Reconstruct the warp field
		std::vector<DualQuaternion> warp_field = m_init_node_se3;
		
		//Query the data
		const float4 can_vertex4 = m_reference_vertex_confid[i];
		const auto knn = m_vertex_knn[i];
		const auto knn_weight = m_vertex_knn_weight[i];
		const float4 depth_vertex_confid = m_reference_vertex_confid[i + offset];
		const float4 depth_normal_radius = m_reference_normal_radius[i + offset];
		
		//Compute analytical gradient
		float residual;
		TwistGradientOfScalarCost gradient;
		device::computePointToPlaneICPTermJacobianResidual(
			depth_vertex_confid,
			depth_normal_radius,
			can_vertex4,
			knn, knn_weight,
			warp_field.data(),
			m_camera2world,
			gradient,
			residual
		);
		
		//Compute the gradient, w.r.t the first knn
		float new_residual;
		TwistGradientOfScalarCost placeholder;
		apply_twist(twist_rot, twist_trans, warp_field[knn.x]);
		device::computePointToPlaneICPTermJacobianResidual(
			depth_vertex_confid,
			depth_normal_radius,
			can_vertex4,
			knn, knn_weight,
			warp_field.data(),
			m_camera2world,
			placeholder,
			new_residual
		);
		
		//The numerical diff
		const auto d_residual_d_nn0_rot0 = (new_residual - residual) / step;
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION)
		const auto analytical_gradient = knn_weight.x * gradient.rotation.x;
#else
		const auto analytical_gradient = (knn_weight.x / (fabsf_sum(knn_weight))) * gradient.rotation.x;
#endif
		const auto diff = d_residual_d_nn0_rot0 - analytical_gradient;
		const auto relative_diff = std::abs(diff / analytical_gradient);
		if(relative_diff > 0.1f && std::abs(d_residual_d_nn0_rot0) > 1e-2) {
			LOG(INFO) << "Relative difference value " << relative_diff << " where the original value is " << analytical_gradient;
		}
	}
}


void surfelwarp::GradientCheck::checkPoint2PointICPJacobian()
{
	//The step
	const float step = 0.01;
	float3 twist_rot = make_float3(step, 0, 0);
	float3 twist_trans = make_float3(0, 0, 0);
	
	const auto offset = 10;
	for(auto i = 0; i < m_reference_vertex_confid.size() - offset; i++) {
		//Reconstruct the warp field
		std::vector<DualQuaternion> warp_field = m_init_node_se3;
		
		//Query the data
		const float4 can_vertex4 = m_reference_vertex_confid[i];
		const auto knn = m_vertex_knn[i];
		const auto knn_weight = m_vertex_knn_weight[i];
		const float4 depth_vertex_confid = m_reference_vertex_confid[i + offset];
		
		//Compute analytical gradient
		float residual[3];
		TwistGradientOfScalarCost gradient[3];
		device::computePointToPointICPTermJacobianResidual(
			depth_vertex_confid,
			can_vertex4,
			knn, knn_weight,
			warp_field.data(),
			m_camera2world,
			residual,
			gradient
		);
		
		//try with new warp field
		float new_residual[3];
		TwistGradientOfScalarCost placeholder[3];
		apply_twist(twist_rot, twist_trans, warp_field[knn.x]);
		device::computePointToPointICPTermJacobianResidual(
			depth_vertex_confid,
			can_vertex4,
			knn, knn_weight,
			warp_field.data(),
			m_camera2world,
			new_residual,
			placeholder
		);
		
		//Compute numerical gradient
		double d_resitual_1_d_nn0_rot_0 = (new_residual[1] - residual[1]) / step;
#if defined(USE_INTERPOLATE_WEIGHT_NORMALIZATION)
		const auto analytical_gradient = knn_weight.x * gradient[1].rotation.x;
#else
		const auto analytical_gradient = (knn_weight.x / (fabsf_sum(knn_weight))) * gradient[1].rotation.x;
#endif
		const auto diff = d_resitual_1_d_nn0_rot_0 - analytical_gradient;
		const auto relative_diff = std::abs(diff / analytical_gradient);
		if(relative_diff > 0.1f && std::abs(d_resitual_1_d_nn0_rot_0) > 1e-2) {
			LOG(INFO) << "Relative difference value " << relative_diff << " where the original value is " << analytical_gradient;
		}
	}
	
}