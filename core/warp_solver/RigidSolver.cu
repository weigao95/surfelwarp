#include "common/device_intrinsics.h"
#include "common/ConfigParser.h"
#include "math/device_mat.h"
#include "core/warp_solver/RigidSolver.h"
#include "RigidSolver.h"

#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {

	struct RigidSolverDevice {

		//The constants for matrix size and blk size
		enum
		{
			//The stoge layout
			lhs_matrix_size = 21,
			rhs_vector_size = 6,
			total_shared_size = lhs_matrix_size + rhs_vector_size,

			//The block size
			block_size = 256,
			num_warps = block_size / 32,
		};
		
		//The map from the renderer
		struct {
			cudaTextureObject_t vertex_map;
			cudaTextureObject_t normal_map;
		} model_maps;

		//The map from the depth image
		struct {
			cudaTextureObject_t vertex_map;
			cudaTextureObject_t normal_map;
		} observation_maps;

		//The camera information
		mat34 init_world2camera;
		Intrinsic intrinsic;

		//The image information
		unsigned image_rows;
		unsigned image_cols;

		//The processing interface
		__device__ __forceinline__ void solverIteration(
			PtrStep<float> reduce_buffer
		) const {
			const auto flatten_pixel_idx = threadIdx.x + blockDim.x * blockIdx.x;
			const auto x = flatten_pixel_idx % image_cols;
			const auto y = flatten_pixel_idx / image_cols;

			//Prepare the jacobian and err
			float jacobian[6] = {0};
			float err = 0.0f;

			//The pixel in range, cannot return as reduction is required
			if(x < image_cols && y < image_rows)
			{
				//Load from the rendered maps
				const float4 model_v4 = tex2D<float4>(model_maps.vertex_map, x, y);
				const float4 model_n4 = tex2D<float4>(model_maps.normal_map, x, y);

				//Transform into camera view
				const auto model_v = init_world2camera.rot * model_v4 + init_world2camera.trans;
				const auto model_n = init_world2camera.rot * model_n4;

				//Project to depth image
				const ushort2 img_coord = {
					__float2uint_rn(((model_v.x / (model_v.z + 1e-10)) * intrinsic.focal_x) + intrinsic.principal_x),
					__float2uint_rn(((model_v.y / (model_v.z + 1e-10)) * intrinsic.focal_y) + intrinsic.principal_y)
				};

				//The projected point is in range
				if(img_coord.x < image_cols && img_coord.y < image_rows)
				{
					//Load the depth map
					const float4 depth_v4 = tex2D<float4>(observation_maps.vertex_map, img_coord.x, img_coord.y);
					const float4 depth_n4 = tex2D<float4>(observation_maps.normal_map, img_coord.x, img_coord.y);

					//Check correspondence
					if(dotxyz(model_n, depth_n4) < 0.8f || squared_distance(model_v, depth_v4) > (0.01f * 0.01f) || is_zero_vertex(depth_v4)) {
						//Pass
					}
					else {
						err = dotxyz(depth_n4, make_float4(model_v.x - depth_v4.x, model_v.y - depth_v4.y, model_v.z - depth_v4.z, 0.0f));
						*(float3*)jacobian = cross_xyz(model_v, depth_n4);
						*(float3*)(jacobian + 3) = make_float3(depth_n4.x, depth_n4.y, depth_n4.z);
					}
				}
			}

			//Time to do reduction
			__shared__ float reduce_mem[total_shared_size][num_warps];
			unsigned shift = 0;
			const auto warp_id = threadIdx.x >> 5;
			const auto lane_id = threadIdx.x & 31;

			//Reduce on matrix
			for (int i = 0; i < 6; i++) { //Row index
				for (int j = i; j < 6; j++) { //Column index, the matrix is symmetry
					float data = (jacobian[i] * jacobian[j]);
					data = warp_scan(data);
					if (lane_id == 31) {
						reduce_mem[shift++][warp_id] = data;
					}
					//Another sync here for reduced mem
					__syncthreads();
				}
			}

			//Reduce on vector
			for (int i = 0; i < 6; i++) {
				float data = (-err * jacobian[i]);
				data = warp_scan(data);
				if (lane_id == 31) {
					reduce_mem[shift++][warp_id] = data;
				}
				//Another sync here for reduced mem
				__syncthreads();
			}

			//Store the result to global memory
			const auto flatten_blk = blockIdx.x + gridDim.x * blockIdx.y;
			for (int i = threadIdx.x; i < total_shared_size; i += 32) {
				if (warp_id == 0) {
					const auto warp_sum = reduce_mem[i][0] + reduce_mem[i][1] + reduce_mem[i][2] + reduce_mem[i][3] 
							+ reduce_mem[i][4] + reduce_mem[i][5] + reduce_mem[i][6] + reduce_mem[i][7];
					reduce_buffer.ptr(i)[flatten_blk] = warp_sum;
				}
			}
		}
	};
	
	__global__ void rigidSolveIterationKernel(
		const RigidSolverDevice solver,
		PtrStep<float> reduce_buffer
	) {
		solver.solverIteration(reduce_buffer);
	}


	__global__ void columnReduceKernel(
		const PtrStepSz<const float> global_buffer,
		float* target
	) {
		const auto idx = threadIdx.x; //There are 32 threads on x direction
		const auto y = threadIdx.y + blockIdx.y * blockDim.y; //There are total memory size on y direction
		float sum = 0.0f;
		for (auto i = threadIdx.x; i < global_buffer.cols; i += 32) {
			sum += global_buffer.ptr(y)[i];
		}

		//__syncthreads();

		// Warp reduction
		sum = warp_scan(sum);
		if (idx == 31) {
			target[y] = sum;
		}
	}



} // device
} // surfelwarp

void surfelwarp::RigidSolver::allocateReduceBuffer() {
	//Allcate the memory for the reduced matrix and vector
	m_reduced_matrix_vector.AllocateBuffer(device::RigidSolverDevice::total_shared_size);
	m_reduced_matrix_vector.ResizeArrayOrException(device::RigidSolverDevice::total_shared_size);
	
	//Allocate the memory for the reduction buffer
	const auto& config = ConfigParser::Instance();
	const auto pixel_size = config.clip_image_rows() * config.clip_image_cols();
	m_reduce_buffer.create(device::RigidSolverDevice::total_shared_size, divUp(pixel_size, device::RigidSolverDevice::block_size));
}

void surfelwarp::RigidSolver::rigidSolveDeviceIteration(cudaStream_t stream) {
	//Construct the device solver
	device::RigidSolverDevice solver;
	
	//The camera info
	solver.intrinsic = m_project_intrinsic;
	solver.init_world2camera = m_curr_world2camera;
	solver.image_rows = m_image_rows;
	solver.image_cols = m_image_cols;
	
	//The map from renderer
	solver.model_maps.vertex_map = m_solver_maps.live_vertex_map;
	solver.model_maps.normal_map = m_solver_maps.live_normal_map;
	
	//The map from observation
	solver.observation_maps.vertex_map = m_observation.vertex_map;
	solver.observation_maps.normal_map = m_observation.normal_map;
	
	dim3 blk(device::RigidSolverDevice::block_size);
	dim3 grid(divUp(m_image_cols * m_image_rows, blk.x));
	device::rigidSolveIterationKernel<<<grid, blk, 0, stream>>>(solver, m_reduce_buffer);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
	
	//Do reduction on the buffer
	device::columnReduceKernel<<<dim3(1, 1, 1), dim3(32, device::RigidSolverDevice::total_shared_size, 1)>>>(
		m_reduce_buffer,
		m_reduced_matrix_vector.DevicePtr()
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif

	//Sync to host
	m_reduced_matrix_vector.SynchronizeToHost(stream, false);
}
