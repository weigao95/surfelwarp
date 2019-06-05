#pragma once
#include "common/common_types.h"
#include "math/vector_ops.hpp"
#include "math/DualQuaternion.hpp"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/solver_types.h"
#include "core/warp_solver/huber_weight.h"

namespace surfelwarp { namespace device {
	

	/**
	 * \brief The dense depth term jacobian and residual, will be used only in handler class
	 */
	__host__ __device__ __forceinline__ void computePointToPlaneICPTermJacobianResidual(
		const float4& depth_vertex_confid,
		const float4& depth_normal_radius,
		const float4& can_vertex4,
		const ushort4& knn, const float4& knn_weight,
		//The warp field
		const DualQuaternion* device_warp_field,
		const mat34& camera2world,
		//The output
		TwistGradientOfScalarCost& twist_graident,
		float& residual
	) {
		//Correct the size
		const float3 can_vertex = make_float3(can_vertex4.x, can_vertex4.y, can_vertex4.z);
		const float3 depth_vertex = make_float3(depth_vertex_confid.x, depth_vertex_confid.y, depth_vertex_confid.z);
		const float3 depth_normal = make_float3(depth_normal_radius.x, depth_normal_radius.y, depth_normal_radius.z);

		//Warp it
		DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
		const mat34 se3 = dq_average.se3_matrix();
		const float3 warped_vertex = se3.rot * can_vertex + se3.trans;

		//Warp the depth vertex to world frame
		const float3 depth_world_normal = camera2world.rot * depth_normal;
		const float3 depth_world_vertex = camera2world.rot * depth_vertex + camera2world.trans;

		//Compute the residual terms
		residual = dot(depth_world_normal, warped_vertex - depth_world_vertex);
		twist_graident.rotation = cross(warped_vertex, depth_world_normal);
		twist_graident.translation = depth_world_normal;
	}


	/**
	 * \brief The method used for computing the feature term jacobian and residuals
	 */
	__host__ __device__ __forceinline__ void computePointToPointICPTermResidual(
		const float4& target_vertex4,
		const float4& warped_vertex,
		//The output
		float residual[3]
	) {
		//Compute the residual
		residual[0] = warped_vertex.x - target_vertex4.x;
		residual[1] = warped_vertex.y - target_vertex4.y;
		residual[2] = warped_vertex.z - target_vertex4.z;
		
		//Apply huber weights
#if defined(USE_FEATURE_HUBER_WEIGHT)
		residual[0] *= compute_feature_huber_weight(residual[0]);
		residual[1] *= compute_feature_huber_weight(residual[1]);
		residual[2] *= compute_feature_huber_weight(residual[2]);
#endif
	}

	__host__ __device__ __forceinline__ void computePointToPointICPTermResidual(
		const float4& target_vertex4,
		const float4& can_vertex4,
		const ushort4& knn, const float4& knn_weight,
		//The warp field
		const DualQuaternion* device_warp_field,
		//The output
		float residual[3]
	) {
		//Correct the size
		const float3 can_vertex = make_float3(can_vertex4.x, can_vertex4.y, can_vertex4.z);

		//Warp it
		DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
		const mat34 se3 = dq_average.se3_matrix();
		const float3 warped_vertex = se3.rot * can_vertex + se3.trans;

		//Handin to processor
		computePointToPointICPTermResidual(target_vertex4, make_float4(warped_vertex.x, warped_vertex.y, warped_vertex.z, 1.0f), residual);
	}


	__host__ __device__ __forceinline__ void computePointToPointICPTermJacobian(
		const float4& target_vertex,
		const float4& warped_vertex,
		//The output
		TwistGradientOfScalarCost* twist_gradient //[3]
	) {
#if defined(USE_FEATURE_HUBER_WEIGHT)
		//Compute the weight
		const float weight_x = compute_feature_huber_weight(warped_vertex.x - target_vertex.x);
		const float weight_y = compute_feature_huber_weight(warped_vertex.y - target_vertex.y);
		const float weight_z = compute_feature_huber_weight(warped_vertex.z - target_vertex.z);

		//Compute the gradient
		twist_gradient[0].translation = weight_x * make_float3(1.0f, 0.0f, 0.0f);
		twist_gradient[0].rotation = weight_x * make_float3(0.0f, warped_vertex.z, -warped_vertex.y);
		twist_gradient[1].translation = weight_y * make_float3(0.0f, 1.0f, 0.0f);
		twist_gradient[1].rotation = weight_y * make_float3(-warped_vertex.z, 0.0f, warped_vertex.x);
		twist_gradient[2].translation = weight_z * make_float3(0.0f, 0.0f, 1.0f);
		twist_gradient[2].rotation = weight_z * make_float3(warped_vertex.y, -warped_vertex.x, 0.0f);

#else
		//Compute the gradient
		twist_gradient[0].translation = make_float3(1.0f, 0.0f, 0.0f);
		twist_gradient[0].rotation = make_float3(0.0f, warped_vertex.z, -warped_vertex.y);
		twist_gradient[1].translation = make_float3(0.0f, 1.0f, 0.0f);
		twist_gradient[1].rotation = make_float3(-warped_vertex.z, 0.0f, warped_vertex.x);
		twist_gradient[2].translation = make_float3(0.0f, 0.0f, 1.0f);
		twist_gradient[2].rotation = make_float3(warped_vertex.y, -warped_vertex.x, 0.0f);
#endif
	}


	__host__ __device__ __forceinline__ void computePointToPointICPTermJacobian(
		const float4& target_vertex,
		const float4& can_vertex4,
		const ushort4& knn, const float4& knn_weight,
		//The warp field
		const DualQuaternion* device_warp_field,
		//The output
		TwistGradientOfScalarCost* twist_gradient //[3]
	) {
		//Correct the size
		const float3 can_vertex = make_float3(can_vertex4.x, can_vertex4.y, can_vertex4.z);
		
		//Warp it
		DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
		const mat34 se3 = dq_average.se3_matrix();
		const float3 warped_vertex = se3.rot * can_vertex + se3.trans;

		//Forward it
		computePointToPointICPTermJacobian(
			target_vertex, 
			make_float4(warped_vertex.x, warped_vertex.y, warped_vertex.z, 1.0f), 
			twist_gradient
		);
	}
	
	__host__ __device__ __forceinline__ void computePoint2PointJtResidual(
		const float4& target_vertex4,
		const float4& warped_vertex,
		float jt_residual[6]
	) {
		//First iter: assign
		float residual = warped_vertex.x - target_vertex4.x;
#if defined(USE_FEATURE_HUBER_WEIGHT)
		float weight = compute_feature_huber_weight(residual);
		residual *= (weight * weight);
#endif
		*((float3*)(&jt_residual[0])) = residual * make_float3(0.0f, warped_vertex.z, -warped_vertex.y);
		*((float3*)(&jt_residual[3])) = residual * make_float3(1.0f, 0.0f, 0.0f);

		//Later iters: plus
		residual = warped_vertex.y - target_vertex4.y;
#if defined(USE_FEATURE_HUBER_WEIGHT)
		weight = compute_feature_huber_weight(residual);
		residual *= (weight * weight);
#endif
		*((float3*)(&jt_residual[0])) += residual * make_float3(-warped_vertex.z, 0.0f, warped_vertex.x);
		*((float3*)(&jt_residual[3])) += residual * make_float3(0.0f, 1.0f, 0.0f);

		residual = warped_vertex.z - target_vertex4.z;
#if defined(USE_FEATURE_HUBER_WEIGHT)
		weight = compute_feature_huber_weight(residual);
		residual *= (weight * weight);
#endif
		*((float3*)(&jt_residual[0])) += residual * make_float3(warped_vertex.y, -warped_vertex.x, 0.0f);
		*((float3*)(&jt_residual[3])) += residual * make_float3(0.0f, 0.0f, 1.0f);
	}


	__host__ __device__ __forceinline__ void computePoint2PointJtResidual(
		const float4& target_vertex4,
		const float4& can_vertex4,
		const ushort4& knn, const float4& knn_weight,
		//The warp field
		const DualQuaternion* device_warp_field,
		//The output
		float jt_residual[6]
	) {
		//Correct the size
		const float3 can_vertex = make_float3(can_vertex4.x, can_vertex4.y, can_vertex4.z);

		//Warp it
		DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
		const mat34 se3 = dq_average.se3_matrix();
		const float3 warped_vertex = se3.rot * can_vertex + se3.trans;

		//Hand in it
		computePoint2PointJtResidual(target_vertex4, make_float4(warped_vertex.x, warped_vertex.y, warped_vertex.z, 1.0f), jt_residual);
	}


	__host__ __device__ __forceinline__ void computePoint2PointJtDot(
		const float4& target_vertex,
		const float4& warped_vertex,
		const float residual[3],
		float jt_residual[6]
	) {
#if defined(USE_FEATURE_HUBER_WEIGHT)
		//Compute the weight
		const float weight_x = compute_feature_huber_weight(warped_vertex.x - target_vertex.x);
		const float weight_y = compute_feature_huber_weight(warped_vertex.y - target_vertex.y);
		const float weight_z = compute_feature_huber_weight(warped_vertex.z - target_vertex.z);

		//First iter: assign
		*((float3*)(&jt_residual[0])) = weight_x * residual[0] * make_float3(0.0f, warped_vertex.z, -warped_vertex.y);
		*((float3*)(&jt_residual[3])) = weight_x * residual[0] * make_float3(1.0f, 0.0f, 0.0f);

		//Later iters: plus
		*((float3*)(&jt_residual[0])) += weight_y * residual[1] * make_float3(-warped_vertex.z, 0.0f, warped_vertex.x);
		*((float3*)(&jt_residual[3])) += weight_y * residual[1] * make_float3(0.0f, 1.0f, 0.0f);

		*((float3*)(&jt_residual[0])) += weight_z * residual[2] * make_float3(warped_vertex.y, -warped_vertex.x, 0.0f);
		*((float3*)(&jt_residual[3])) += weight_z * residual[2] * make_float3(0.0f, 0.0f, 1.0f);
#else
		//First iter: assign
		*((float3*)(&jt_residual[0])) = residual[0] * make_float3(0.0f, warped_vertex.z, -warped_vertex.y);
		*((float3*)(&jt_residual[3])) = residual[0] * make_float3(1.0f, 0.0f, 0.0f);

		//Later iters: plus
		*((float3*)(&jt_residual[0])) += residual[1] * make_float3(-warped_vertex.z, 0.0f, warped_vertex.x);
		*((float3*)(&jt_residual[3])) += residual[1] * make_float3(0.0f, 1.0f, 0.0f);

		*((float3*)(&jt_residual[0])) += residual[2] * make_float3(warped_vertex.y, -warped_vertex.x, 0.0f);
		*((float3*)(&jt_residual[3])) += residual[2] * make_float3(0.0f, 0.0f, 1.0f);
#endif		
	}

	__host__ __device__ __forceinline__ void computePoint2PointJtDot(
		const float4& target_vertex,
		const float4& can_vertex4,
		const ushort4& knn, const float4& knn_weight,
		//The warp field
		const DualQuaternion* device_warp_field,
		const float residual[3],
		//The output
		float jt_residual[6]
	) {
		//Correct the size
		const float3 can_vertex = make_float3(can_vertex4.x, can_vertex4.y, can_vertex4.z);

		//Warp it
		DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
		const mat34 se3 = dq_average.se3_matrix();
		const float3 warped_vertex = se3.rot * can_vertex + se3.trans;

		//Hand in it
		computePoint2PointJtDot(target_vertex, make_float4(warped_vertex.x, warped_vertex.y, warped_vertex.z, 1.0f), residual, jt_residual);
	}


	/**
	 * \brief The method to compute the jacobian and residual of smooth term.
	 *        The underlying form is the same, the variants are for efficiency
	 */
	__host__ __forceinline__ void computeSmoothTermJacobianResidual(
		const float4& xi4, const float4& xj4,
		const mat34& Ti, const mat34& Tj,
		float* residual, // [3]
		TwistGradientOfScalarCost* twist_gradient_i,//[3]
		TwistGradientOfScalarCost* twist_gradient_j //[3]
	) {
		const float3 xi = make_float3(xi4.x, xi4.y, xi4.z);
		const float3 xj = make_float3(xj4.x, xj4.y, xj4.z);
		const float3 r = Ti.rot * xj + Ti.trans;
		const float3 s = Tj.rot * xj + Tj.trans;

		//Compute the residual
		residual[0] = r.x - s.x;
		residual[1] = r.y - s.y;
		residual[2] = r.z - s.z;

		//Compute the jacobian
		twist_gradient_i[0].rotation = make_float3(0.0f, r.z, -r.y);
		twist_gradient_i[1].rotation = make_float3(-r.z, 0.0f, r.x);
		twist_gradient_i[2].rotation = make_float3(r.y, -r.x, 0.0f);
		twist_gradient_i[0].translation = make_float3(1.0f, 0.0f, 0.0f);
		twist_gradient_i[1].translation = make_float3(0.0f, 1.0f, 0.0f);
		twist_gradient_i[2].translation = make_float3(0.0f, 0.0f, 1.0f);

		twist_gradient_j[0].rotation = make_float3(0.0f, -s.z, s.y);
		twist_gradient_j[1].rotation = make_float3(s.z, 0.0f, -s.x);
		twist_gradient_j[2].rotation = make_float3(-s.y, s.x, 0.0f);
		twist_gradient_j[0].translation = make_float3(-1.0f,  0.0f,  0.0f);
		twist_gradient_j[1].translation = make_float3( 0.0f, -1.0f,  0.0f);
		twist_gradient_j[2].translation = make_float3( 0.0f,  0.0f, -1.0f);
	}


	__host__ __device__ __forceinline__ void computeSmoothTermResidual(
		const float3& Ti_xj,
		const float3& Tj_xj,
		float* residual // [3]
	) {
		residual[0] = Ti_xj.x - Tj_xj.x;
		residual[1] = Ti_xj.y - Tj_xj.y;
		residual[2] = Ti_xj.z - Tj_xj.z;
	}

	__host__ __device__ __forceinline__ void computeSmoothTermResidual(
		const float4& xj4,
		const mat34& Ti, const mat34& Tj,
		float* residual // [3]
	) {
		const float3 xj = make_float3(xj4.x, xj4.y, xj4.z);
		const float3 r = Ti.rot * xj + Ti.trans;
		const float3 s = Tj.rot * xj + Tj.trans;

		//Compute the residual
		computeSmoothTermResidual(r, s, residual);
	}

	__host__ __device__ __forceinline__ void computeSmoothTermJacobian(
		const float3& Ti_xj,
		const float3& Tj_xj,
		TwistGradientOfScalarCost* twist_gradient_i,//[3]
		TwistGradientOfScalarCost* twist_gradient_j //[3]
	) {
		//Compute the jacobian
		twist_gradient_i[0].rotation = make_float3(0.0f, Ti_xj.z, -Ti_xj.y);
		twist_gradient_i[0].translation = make_float3(1.0f, 0.0f, 0.0f);
		twist_gradient_i[1].rotation = make_float3(-Ti_xj.z, 0.0f, Ti_xj.x);
		twist_gradient_i[1].translation = make_float3(0.0f, 1.0f, 0.0f);
		twist_gradient_i[2].rotation = make_float3(Ti_xj.y, -Ti_xj.x, 0.0f);
		twist_gradient_i[2].translation = make_float3(0.0f, 0.0f, 1.0f);

		twist_gradient_j[0].rotation = make_float3(0.0f, -Tj_xj.z, Tj_xj.y);
		twist_gradient_j[0].translation = make_float3(-1.0f,  0.0f,  0.0f);
		twist_gradient_j[1].rotation = make_float3(Tj_xj.z, 0.0f, -Tj_xj.x);
		twist_gradient_j[1].translation = make_float3( 0.0f, -1.0f,  0.0f);
		twist_gradient_j[2].rotation = make_float3(-Tj_xj.y, Tj_xj.x, 0.0f);
		twist_gradient_j[2].translation = make_float3( 0.0f,  0.0f, -1.0f);
	}

	__host__ __device__ __forceinline__ void computeSmoothTermJacobian(
		const float4& xj4,
		const mat34& Ti, const mat34& Tj,
		TwistGradientOfScalarCost* twist_gradient_i,//[3]
		TwistGradientOfScalarCost* twist_gradient_j //[3]
	) {
		const float3 xj = make_float3(xj4.x, xj4.y, xj4.z);
		const float3 r = Ti.rot * xj + Ti.trans;
		const float3 s = Tj.rot * xj + Tj.trans;

		//Compute the jacobian
		computeSmoothTermJacobian(r, s, twist_gradient_i, twist_gradient_j);
	}

	__host__ __device__ __forceinline__ void computeSmoothTermJacobian(
		const float3& Ti_xj, const float3& Tj_xj,
		bool is_node_i,
		TwistGradientOfScalarCost* twist_gradient//[3]
	) {
		if(is_node_i)
		{
			twist_gradient[0].rotation = make_float3(0.0f, Ti_xj.z, -Ti_xj.y);
			twist_gradient[0].translation = make_float3(1.0f, 0.0f, 0.0f);
			twist_gradient[1].rotation = make_float3(-Ti_xj.z, 0.0f, Ti_xj.x);
			twist_gradient[1].translation = make_float3(0.0f, 1.0f, 0.0f);
			twist_gradient[2].rotation = make_float3(Ti_xj.y, -Ti_xj.x, 0.0f);
			twist_gradient[2].translation = make_float3(0.0f, 0.0f, 1.0f);
		}
		else
		{
			twist_gradient[0].rotation = make_float3(0.0f, -Tj_xj.z, Tj_xj.y);
			twist_gradient[0].translation = make_float3(-1.0f,  0.0f,  0.0f);
			twist_gradient[1].rotation = make_float3(Tj_xj.z, 0.0f, -Tj_xj.x);
			twist_gradient[1].translation = make_float3( 0.0f, -1.0f,  0.0f);
			twist_gradient[2].rotation = make_float3(-Tj_xj.y, Tj_xj.x, 0.0f);
			twist_gradient[2].translation = make_float3( 0.0f,  0.0f, -1.0f);
		}
	}

	__host__ __device__ __forceinline__ void computeSmoothTermJacobian(
		const float4& xj4,
		const mat34& Ti, const mat34& Tj,
		bool is_node_i,
		TwistGradientOfScalarCost* twist_gradient//[3]
	) {
		const float3 xj = make_float3(xj4.x, xj4.y, xj4.z);
		const float3 r = Ti.rot * xj + Ti.trans;
		const float3 s = Tj.rot * xj + Tj.trans;

		
		computeSmoothTermJacobian(r, s, is_node_i, twist_gradient);
	}


	__host__ __device__ __forceinline__ void computeSmoothTermJtResidual(
		const float3& Ti_xj,
		const float3& Tj_xj,
		bool is_node_i,
		float jt_residual[6]
	) {
		if(is_node_i)
		{
			//First iter: assign
			float residual = Ti_xj.x - Tj_xj.x;
			*((float3*)(&jt_residual[0])) = residual * make_float3(0.0f, Ti_xj.z, -Ti_xj.y);
			*((float3*)(&jt_residual[3])) = residual * make_float3(1.0f, 0.0f, 0.0f);

			//Next iters: plus
			residual = Ti_xj.y - Tj_xj.y;
			*((float3*)(&jt_residual[0])) += residual * make_float3(-Ti_xj.z, 0.0f, Ti_xj.x);
			*((float3*)(&jt_residual[3])) += residual * make_float3(0.0f, 1.0f, 0.0f);

			residual = Ti_xj.z - Tj_xj.z;
			*((float3*)(&jt_residual[0])) += residual * make_float3(Ti_xj.y, -Ti_xj.x, 0.0f);
			*((float3*)(&jt_residual[3])) += residual * make_float3(0.0f, 0.0f, 1.0f);
		}
		else
		{
			//First iter: assign
			float residual = Ti_xj.x - Tj_xj.x;
			*((float3*)(&jt_residual[0])) = residual * make_float3(0.0f, -Tj_xj.z, Tj_xj.y);
			*((float3*)(&jt_residual[3])) = residual * make_float3(-1.0f,  0.0f,  0.0f);

			//Next iters: plus
			residual = Ti_xj.y - Tj_xj.y;
			*((float3*)(&jt_residual[0])) += residual * make_float3(Tj_xj.z, 0.0f, -Tj_xj.x);
			*((float3*)(&jt_residual[3])) += residual * make_float3(0.0f, -1.0f, 0.0f);

			residual = Ti_xj.z - Tj_xj.z;
			*((float3*)(&jt_residual[0])) += residual * make_float3(-Tj_xj.y, Tj_xj.x, 0.0f);
			*((float3*)(&jt_residual[3])) += residual * make_float3(0.0f, 0.0f, -1.0f);
		}
	}

	__host__ __device__ __forceinline__ void computeSmoothTermJtResidual(
		const float4& xj4,
		const mat34& Ti, const mat34& Tj,
		bool is_node_i,
		float jt_residual[6]
	) {
		const float3 xj = make_float3(xj4.x, xj4.y, xj4.z);
		const float3 r = Ti.rot * xj + Ti.trans;
		const float3 s = Tj.rot * xj + Tj.trans;

		//Combine the computation
		computeSmoothTermJtResidual(r, s, is_node_i, jt_residual);
	}


	__host__ __device__ __forceinline__ void computeSmoothTermJtDot(
		const float3& Ti_xj, const float3& Tj_xj,
		bool is_node_i,
		const float residual[3],
		float jt_residual[6]
	) {
		if(is_node_i)
		{
			//First iter: assign
			*((float3*)(&jt_residual[0])) = residual[0] * make_float3(0.0f, Ti_xj.z, -Ti_xj.y);
			*((float3*)(&jt_residual[3])) = residual[0] * make_float3(1.0f, 0.0f, 0.0f);

			//Next iters: plus
			*((float3*)(&jt_residual[0])) += residual[1] * make_float3(-Ti_xj.z, 0.0f, Ti_xj.x);
			*((float3*)(&jt_residual[3])) += residual[1] * make_float3(0.0f, 1.0f, 0.0f);

			*((float3*)(&jt_residual[0])) += residual[2] * make_float3(Ti_xj.y, -Ti_xj.x, 0.0f);
			*((float3*)(&jt_residual[3])) += residual[2] * make_float3(0.0f, 0.0f, 1.0f);
		}
		else
		{
			//First iter: assign
			*((float3*)(&jt_residual[0])) = residual[0] * make_float3(0.0f, -Tj_xj.z, Tj_xj.y);
			*((float3*)(&jt_residual[3])) = residual[0] * make_float3(-1.0f,  0.0f,  0.0f);

			//Next iters: plus
			*((float3*)(&jt_residual[0])) += residual[1] * make_float3(Tj_xj.z, 0.0f, -Tj_xj.x);
			*((float3*)(&jt_residual[3])) += residual[1] * make_float3(0.0f, -1.0f, 0.0f);

			*((float3*)(&jt_residual[0])) += residual[2] * make_float3(-Tj_xj.y, Tj_xj.x, 0.0f);
			*((float3*)(&jt_residual[3])) += residual[2] * make_float3(0.0f, 0.0f, -1.0f);
		}
	}

	__host__ __device__ __forceinline__ void computeSmoothTermJtDot(
		const float4& xj4,
		const mat34& Ti, const mat34& Tj,
		bool is_node_i,
		const float residual[3],
		float jt_residual[6]
	) {
		const float3 xj = make_float3(xj4.x, xj4.y, xj4.z);
		const float3 r = Ti.rot * xj + Ti.trans;
		const float3 s = Tj.rot * xj + Tj.trans;

		//Combine the computation
		computeSmoothTermJtDot(r ,s, is_node_i, residual, jt_residual);
	}


} // namespace device
} // namespace surfelwarp
