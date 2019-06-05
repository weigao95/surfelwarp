#include "imgproc/generate_maps.h"
#include "math/vector_ops.hpp"
#include "math/eigen33.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	//Note the depth value are in meters
	//The computed radius are in [mm]
	__host__ __device__ __forceinline__ 
	float compute_radius(
		float depth_value, // [m]
		float normal_z
	) {
		const float radius = depth_value * (1000 * 1.414f / 570.0f);
		normal_z = abs(normal_z);
		float radius_n = 2.0f * radius;
		if (normal_z > 0.5) {
			radius_n = radius / normal_z;
		}
		return radius_n;//[mm]
	}

	enum {
		window_dim = 7,
		halfsize = 3,
		window_size = window_dim * window_dim
	};


	__global__ void createNormalRadiusMapKernel(
		cudaTextureObject_t vertex_map,
		const unsigned rows, const unsigned cols,
		cudaSurfaceObject_t normal_radius_map
	){
		const auto x = threadIdx.x + blockDim.x*blockIdx.x;
		const auto y = threadIdx.y + blockDim.y*blockIdx.y;
		if (x >= cols || y >= rows) return;

		//This value must be written to surface at end
		float4 normal_radius_value = make_float4(0, 0, 0, 0);

		//The vertex at the center
		const float4 vertex_center = tex2D<float4>(vertex_map, x, y);
		if (!is_zero_vertex(vertex_center)) {
			float4 centeroid = make_float4(0, 0, 0, 0);
			int counter = 0;
			//First window search to determine the center
			for (int cy = y - halfsize; cy <= y + halfsize; cy += 1) {
				for (int cx = x - halfsize; cx <= x + halfsize; cx += 1) {
					const float4 p = tex2D<float4>(vertex_map, cx, cy);
					if (!is_zero_vertex(p)) {
						centeroid.x += p.x;
						centeroid.y += p.y;
						centeroid.z += p.z;
						counter++;
					}
				}
			}//End of first window search

			//At least half of the window is valid
			if(counter > (window_size / 2)) {
				centeroid *= (1.0f / counter);
				float covariance[6] = { 0 };

				//Second window search to compute the normal
				for (int cy = y - halfsize; cy < y + halfsize; cy += 1) {
					for (int cx = x - halfsize; cx < x + halfsize; cx += 1) {
						const float4 p = tex2D<float4>(vertex_map, cx, cy);
						if (!is_zero_vertex(p)) {
							const float4 diff = p - centeroid;
							//Compute the covariance
							covariance[0] += diff.x * diff.x; //(0, 0)
							covariance[1] += diff.x * diff.y; //(0, 1)
							covariance[2] += diff.x * diff.z; //(0, 2)
							covariance[3] += diff.y * diff.y; //(1, 1)
							covariance[4] += diff.y * diff.z; //(1, 2)
							covariance[5] += diff.z * diff.z; //(2, 2)
						}
					}
				}//End of second window search

				//The eigen value for normal
				eigen33 eigen(covariance);
				float3 normal;
				eigen.compute(normal);
				if (dotxyz(normal, vertex_center) >= 0.0f) normal *= -1;

				//The radius
				const float radius = compute_radius(vertex_center.z, normal.z);

				//Write to local variable
				normal_radius_value.x = normal.x;
				normal_radius_value.y = normal.y;
				normal_radius_value.z = normal.z;
				normal_radius_value.w = radius;
			}//End of check the number of valid pixels
		}//If the vertex is non-zero
		
		//Write to the surface
		surf2Dwrite(normal_radius_value, normal_radius_map, x * sizeof(float4), y);
	}

}; /* End of namespace device */
}; /* End of namespace surfelwarp */


void surfelwarp::createNormalRadiusMap(
	cudaTextureObject_t vertex_map, 
	const unsigned rows, const unsigned cols, 
	cudaSurfaceObject_t normal_radius_map, 
	cudaStream_t stream
) {
	dim3 blk(16, 16);
	dim3 grid(divUp(cols, blk.x), divUp(rows, blk.y));
	device::createNormalRadiusMapKernel<<<grid, blk, 0, stream>>>(
		vertex_map, 
		rows, cols, 
		normal_radius_map
	);

	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}