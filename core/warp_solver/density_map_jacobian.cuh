#pragma once
#include "common/common_types.h"
#include "common/color_transfer.h"
#include "math/DualQuaternion.hpp"
#include "core/warp_solver/solver_types.h"
#include "core/warp_solver/huber_weight.h"

namespace surfelwarp { namespace device {

    __device__ __forceinline__ void computeImageDensityJacobainAndResidual(
		//The maps from camera observation
		cudaTextureObject_t density_map, // float1 texture
		cudaTextureObject_t density_gradient_map, // float2 texture
		unsigned width, unsigned height,
		//The queried vertex, knn and weight from pixel
		const float4& reference_vertex,
		const float geometry_density,
		const ushort4& knn,
		const float4& knn_weight,
		//The warp field information
		const DualQuaternion* device_warp_field,
		const Intrinsic& intrinsic, const mat34& world2camera,
		//Output the jacobian and rhs
		TwistGradientOfScalarCost& twist_gradient, 
		float& rhs
	) {
		//Warp the reference vertex
		const float3 can_vertex = make_float3(reference_vertex.x, reference_vertex.y, reference_vertex.z);
		DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
		const mat34 se3 = dq_average.se3_matrix();
		const float3 warped_vertex = se3.rot * can_vertex + se3.trans;

		//Transfer to camera frame and project to pixel
		const float3 warped_vertex_camera = world2camera.rot * warped_vertex + world2camera.trans;
		const int2 img_coord = {
			__float2int_rn(((warped_vertex_camera.x / (warped_vertex_camera.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x),
			__float2int_rn(((warped_vertex_camera.y / (warped_vertex_camera.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y)
		};


		if(img_coord.x >= 0 && img_coord.x < width && img_coord.y >= 0 && img_coord.y < height)
		{
			//The density value
			const float image_density = tex2D<float>(density_map, img_coord.x, img_coord.y);
			const float2 image_density_gradient = tex2D<float2>(density_gradient_map, img_coord.x, img_coord.y);

			//The jacobian of the density value to the camera frame position
			float3 d_density_d_position; // Image it as a row vector
			d_density_d_position.x = image_density_gradient.x * intrinsic.focal_x / warped_vertex_camera.z;
			d_density_d_position.y = image_density_gradient.y * intrinsic.focal_y / warped_vertex_camera.z;
			d_density_d_position.z = -1.0f/(warped_vertex_camera.z*warped_vertex_camera.z) * (intrinsic.focal_x*warped_vertex_camera.x*image_density_gradient.x + intrinsic.focal_y*warped_vertex_camera.y*image_density_gradient.y);

			//Transfer to world position by dotting with the rotation part of the camera2world matrix
			d_density_d_position = world2camera.rot.transpose_dot(d_density_d_position);

			//Evaluate the jacobian towards the twist
			twist_gradient.rotation.x = (-d_density_d_position.y * warped_vertex.z + d_density_d_position.z * warped_vertex.y);
			twist_gradient.rotation.y = ( d_density_d_position.x * warped_vertex.z - d_density_d_position.z * warped_vertex.x);
			twist_gradient.rotation.z = (-d_density_d_position.x * warped_vertex.y + d_density_d_position.y * warped_vertex.x);
			twist_gradient.translation = d_density_d_position;

			//The rhs is just the difference between rendered and real density
			rhs = image_density - geometry_density;
		}
		else //The pixel is outside the image, jacobian is zero
		{
			twist_gradient.rotation.x = twist_gradient.rotation.y = twist_gradient.rotation.z = 0.0f;
			twist_gradient.translation.x = twist_gradient.translation.y = twist_gradient.translation.z = 0.0f;
			rhs = 0.0f;
		}
	}



	__device__ __forceinline__ void computeImageDensityJacobainAndResidualHuberWeight(
		//The maps from camera observation
		cudaTextureObject_t density_map, // float1 texture
		cudaTextureObject_t density_gradient_map, // float2 texture
		unsigned width, unsigned height,
		//The queried vertex, knn and weight from pixel
		const float4& reference_vertex,
		const float geometry_density,
		const ushort4& knn,
		const float4& knn_weight,
		//The warp field information
		const DualQuaternion* device_warp_field,
		const Intrinsic& intrinsic, const mat34& world2camera,
		//Output the jacobian and rhs
		TwistGradientOfScalarCost& twist_gradient, 
		float& rhs,
		float huber_cutoff = 1.0f
	) {
		//Warp the reference vertex
		const float3 can_vertex = make_float3(reference_vertex.x, reference_vertex.y, reference_vertex.z);
		DualQuaternion dq_average = averageDualQuaternion(device_warp_field, knn, knn_weight);
		const mat34 se3 = dq_average.se3_matrix();
		const float3 warped_vertex = se3.rot * can_vertex + se3.trans;

		//Transfer to camera frame and project to pixel
		const float3 warped_vertex_camera = world2camera.rot * warped_vertex + world2camera.trans;
		const int2 img_coord = {
			__float2int_rn(((warped_vertex_camera.x / (warped_vertex_camera.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x),
			__float2int_rn(((warped_vertex_camera.y / (warped_vertex_camera.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y)
		};


		if(img_coord.x >= 0 && img_coord.x < width && img_coord.y >= 0 && img_coord.y < height)
		{
			//The density value
			const float image_density = tex2D<float>(density_map, img_coord.x, img_coord.y);
			const float2 image_density_gradient = tex2D<float2>(density_gradient_map, img_coord.x, img_coord.y);

			//The jacobian of the density value to the camera frame position
			float3 d_density_d_position; // Image it as a row vector
			d_density_d_position.x = image_density_gradient.x * intrinsic.focal_x / warped_vertex_camera.z;
			d_density_d_position.y = image_density_gradient.y * intrinsic.focal_y / warped_vertex_camera.z;
			d_density_d_position.z = -1.0f/(warped_vertex_camera.z*warped_vertex_camera.z) * (intrinsic.focal_x*warped_vertex_camera.x*image_density_gradient.x + intrinsic.focal_y*warped_vertex_camera.y*image_density_gradient.y);

			//Transfer to world position by dotting with the rotation part of the camera2world matrix
			d_density_d_position = world2camera.rot.transpose_dot(d_density_d_position);

			//The rhs is just the difference between rendered and real density
			const float huber_weight = compute_huber_weight(image_density - geometry_density, huber_cutoff);
			rhs = huber_weight * (image_density - geometry_density);

			//Evaluate the jacobian towards the twist
			twist_gradient.rotation.x = huber_weight * (-d_density_d_position.y * warped_vertex.z + d_density_d_position.z * warped_vertex.y);
			twist_gradient.rotation.y = huber_weight * ( d_density_d_position.x * warped_vertex.z - d_density_d_position.z * warped_vertex.x);
			twist_gradient.rotation.z = huber_weight * (-d_density_d_position.x * warped_vertex.y + d_density_d_position.y * warped_vertex.x);
			twist_gradient.translation = huber_weight * d_density_d_position;
		}
		else //The pixel is outside the image, jacobian is zero
		{
			twist_gradient.rotation.x = twist_gradient.rotation.y = twist_gradient.rotation.z = 0.0f;
			twist_gradient.translation.x = twist_gradient.translation.y = twist_gradient.translation.z = 0.0f;
			rhs = 0.0f;
		}
	}


} // namespace device
} // namespace surfelwarp