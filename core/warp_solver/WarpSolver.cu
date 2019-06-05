#include "core/warp_solver/WarpSolver.h"
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__global__ void queryPixelKNNKernel(
		cudaTextureObject_t index_map,
		const ushort4* surfel_knn,
		const float4* surfel_knn_weight,
		//Output
		PtrStepSz<KNNAndWeight> knn_map
	) {
		const auto x = threadIdx.x + blockDim.x * blockIdx.x;
		const auto y = threadIdx.y + blockDim.y * blockIdx.y;
		if(x < knn_map.cols && y < knn_map.rows)
		{
			KNNAndWeight knn_weight;
			knn_weight.set_invalid();
			const auto index = tex2D<unsigned>(index_map, x, y);
			if(index != 0xFFFFFFFF) {
				knn_weight.knn = surfel_knn[index];
				knn_weight.weight = surfel_knn_weight[index];
			}
			
			//Store to result
			knn_map.ptr(y)[x] = knn_weight;
		}
	}


} // namespace device
} // namespace surfelwarp



void surfelwarp::WarpSolver::QueryPixelKNN(cudaStream_t stream) {
	dim3 blk(16, 16);
	dim3 grid(divUp(m_knn_map.cols(), blk.x), divUp(m_knn_map.rows(), blk.y));
	device::queryPixelKNNKernel<<<grid, blk, 0, stream>>>(
		m_rendered_maps.index_map,
		m_geometry_input.surfel_knn,
		m_geometry_input.surfel_knn_weight,
		m_knn_map
	);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}


/* The method to setup and solve Ax=b using pcg solver
 */
void surfelwarp::WarpSolver::allocatePCGSolverBuffer() {
	const auto max_matrix_size = 6 * Constants::kMaxNumNodes;
	m_pcg_solver = std::make_shared<BlockPCG<6>>(max_matrix_size);
}

void surfelwarp::WarpSolver::releasePCGSolverBuffer() {
}

void surfelwarp::WarpSolver::UpdatePCGSolverStream(cudaStream_t stream) {
	m_pcg_solver->UpdateCudaStream(stream);
}

void surfelwarp::WarpSolver::SolvePCGMatrixFree() {
	//Prepare the data
	const auto inversed_diagonal_preconditioner = m_preconditioner_rhs_builder->InversedPreconditioner();
	const auto rhs = m_preconditioner_rhs_builder->JtDotResidualValue();
	ApplySpMVBase<6>::Ptr apply_spmv_handler = m_apply_jtj_handler;
	DeviceArraySlice<float> updated_twist = m_iteration_data.CurrentWarpFieldUpdateBuffer();
	
	//sanity check
	SURFELWARP_CHECK_EQ(rhs.Size(), apply_spmv_handler->MatrixSize());
	SURFELWARP_CHECK_EQ(updated_twist.Size(), apply_spmv_handler->MatrixSize());
	SURFELWARP_CHECK_EQ(inversed_diagonal_preconditioner.Size(), apply_spmv_handler->MatrixSize() * 6);
	
	//Hand in to warp solver and solve it
	m_pcg_solver->SetSolverInput(inversed_diagonal_preconditioner, apply_spmv_handler, rhs, updated_twist);
	m_pcg_solver->Solve(10);
}

void surfelwarp::WarpSolver::SolvePCGMaterialized(int pcg_iterations) {
	//Prepare the data
	const auto inversed_diagonal_preconditioner = m_preconditioner_rhs_builder->InversedPreconditioner();
	const auto rhs = m_preconditioner_rhs_builder->JtDotResidualValue();
	ApplySpMVBase<6>::Ptr apply_spmv_handler = m_jtj_materializer->GetSpMVHandler();
	DeviceArraySlice<float> updated_twist = m_iteration_data.CurrentWarpFieldUpdateBuffer();
	
	//sanity check
	SURFELWARP_CHECK_EQ(rhs.Size(), apply_spmv_handler->MatrixSize());
	SURFELWARP_CHECK_EQ(updated_twist.Size(), apply_spmv_handler->MatrixSize());
	SURFELWARP_CHECK_EQ(inversed_diagonal_preconditioner.Size(), apply_spmv_handler->MatrixSize() * 6);
	
	//Hand in to warp solver and solve it
	m_pcg_solver->SetSolverInput(inversed_diagonal_preconditioner, apply_spmv_handler, rhs, updated_twist);
	m_pcg_solver->Solve(pcg_iterations);
}