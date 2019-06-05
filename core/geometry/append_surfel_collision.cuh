#pragma once

#include "math/vector_ops.hpp"
#include "math/DualQuaternion.hpp"
#include <math_functions.h>

namespace surfelwarp {
	
	__host__ __device__ __forceinline__ bool is_compressive_mapped(
		const float4& vertex,
		const ushort4& vertex_knn, const float4& vertex_knnweight,
		const float4* finitediff_vertex,
		const ushort4* finitediff_knn,
		const float4* finitediff_weight,
		const DualQuaternion* node_se3,
		const float finite_step
	) {
		//First compute the warp-back position of the vertex
		DualQuaternion dq_average = averageDualQuaternion(node_se3, vertex_knn, vertex_knnweight);
		mat34 se3_vertex = dq_average.se3_matrix();
		const float3 reference_vertex = se3_vertex.apply_inversed_se3(vertex);
		const float* ref_vertex_flatten = (const float*)(&reference_vertex);

		//Iterave through every directions to compute volumn strain
		float volumn_strain = 0.0f;
		for(auto i = 0; i < 3; i++) {
			dq_average = averageDualQuaternion(node_se3, finitediff_knn[i], finitediff_weight[i]);
			se3_vertex = dq_average.se3_matrix();
			//Perform an inverse warping here
			const float3 ref_finitediff_vertex = se3_vertex.apply_inversed_se3(finitediff_vertex[i]);
			const float* ref_finitediff_vertex_flatten = (const float*)(&ref_finitediff_vertex);
			const float dx = ref_finitediff_vertex_flatten[i] - ref_vertex_flatten[i] - finite_step;
			if(dx > 0.0f) // Only count for extension?
				volumn_strain += (dx / finite_step);
		}

		//Check it
		if(volumn_strain > 0.3f) return true;
		return false;
	}

} // surfelwarp