#include "core/warp_solver/WarpSolver.h"

//The input interface
void surfelwarp::WarpSolver::SetSolverInputs(
	CameraObservation observation,
	Renderer::SolverMaps rendered_maps,
	SurfelGeometry::SolverInput geometry_input,
	WarpField::SolverInput warpfield_input,
	const mat34 &world2camera
) {
	m_observation = observation;
	m_rendered_maps = rendered_maps;
	m_geometry_input = geometry_input;
	m_warpfield_input = warpfield_input;
	m_world2camera = world2camera;
	
	//The iteration data
	m_iteration_data.SetWarpFieldInitialValue(warpfield_input.node_se3);
}


void surfelwarp::WarpSolver::SetSolverInputs(
	CameraObservation observation,
	Renderer::SolverMaps rendered_maps,
	SurfelGeometry::SolverInput geometry_input,
	WarpField::SolverInput warpfield_input,
	const Matrix4f& world2camera
) {
	SetSolverInputs(
		observation,
		rendered_maps,
		geometry_input,
		warpfield_input,
		mat34(world2camera)
	);
}


//The global serial solver interface
void surfelwarp::WarpSolver::SolveSerial(cudaStream_t stream) {
	//solveMatrixFreeFixedIndexSerial(stream);
	//solveMaterializedFixedIndexSerial(stream);
	solveMaterializedFixedIndexGlobalLocalSerial(stream);
}

//The matrix-free fixed index solver
void surfelwarp::WarpSolver::solveMatrixFreeFixedIndexSerial(cudaStream_t stream) {
	QueryPixelKNN(stream);
	fullSolverIterationMatrixFreeFixedIndexSerial(stream);
	for (auto i = 0; i < Constants::kNumGaussNewtonIterations - 1; i++) {
		matrixFreeFixedIndexSolverIterationSerial(stream);
	}
}

void surfelwarp::WarpSolver::fullSolverIterationMatrixFreeFixedIndexSerial(cudaStream_t stream) {
	FetchPotentialDenseImageTermPixelsFixedIndexSynced(stream);
	setDenseDepthHandlerFullInput();
	setDensityForegroundHandlerFullInput();
	FindPotentialForegroundMaskPixelSynced(stream);
	//FindValidColorForegroundMaskPixel(stream, stream);
	SelectValidSparseFeatureMatchedPairs(stream);

	//The index part
	SetNode2TermIndexInput();
	BuildNode2TermIndex(stream);

	//Compute jacobian
	ComputeTermJacobianFixedIndex(stream, stream, stream, stream);

	//The computation of diagonal blks JtJ and JtError
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditioner(stream);
	ComputeJtResidual(stream);

	//Debug methods
	LOG(INFO) << "The total squared residual is " << ComputeTotalResidualSynced(stream);

	//Solve it and update
	SolvePCGMatrixFree();
	m_iteration_data.ApplyWarpFieldUpdate(stream);
}

void surfelwarp::WarpSolver::matrixFreeFixedIndexSolverIterationSerial(cudaStream_t stream) {
	//Hand in the new SE3 to handlers
	m_dense_depth_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_density_foreground_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_sparse_correspondence_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());

	//Re-compute the jacobian
	ComputeTermJacobianFixedIndex(stream, stream, stream, stream);
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditioner(stream);
	ComputeJtResidual(stream);

	//Debug methods
	LOG(INFO) << "The total squared residual is " << ComputeTotalResidualSynced(stream);

	//Solve it and update
	SolvePCGMatrixFree();
	m_iteration_data.ApplyWarpFieldUpdate(stream);
}

//The solver interface for materialized, fixed index solver
void surfelwarp::WarpSolver::solveMaterializedFixedIndexSerial(cudaStream_t stream) {
	QueryPixelKNN(stream);
	fullSolverIterationMaterializedFixedIndexSerial(stream);
	for(auto i = 0; i < Constants::kNumGaussNewtonIterations; i++) {
		materializedFixedIndexSolverIterationSerial(stream);
	}
}

void surfelwarp::WarpSolver::fullSolverIterationMaterializedFixedIndexSerial(cudaStream_t stream) {
	FetchPotentialDenseImageTermPixelsFixedIndexSynced(stream);
	setDenseDepthHandlerFullInput();
	setDensityForegroundHandlerFullInput();
	FindPotentialForegroundMaskPixelSynced(stream);
	//FindValidColorForegroundMaskPixel(stream, stream);
	SelectValidSparseFeatureMatchedPairs(stream);
	
	//The indexing part
	SetNode2TermIndexInput();
	BuildNode2TermIndex(stream);
	BuildNodePair2TermIndexBlocked(stream);
	
	//The computation of jacobian
	ComputeTermJacobianFixedIndex(stream, stream, stream, stream);
	
	//The computation of diagonal blks JtJ and JtError
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditioner(stream);
	ComputeJtResidual(stream);
	
	//The non-diagonal blks of the matrix
	SetJtJMaterializerInput();
	MaterializeJtJNondiagonalBlocks(stream);
	
	//The assemble of matrix
	const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, stream);
	
	//Debug methods
	LOG(INFO) << "The total squared residual in materialized, fixed-index solver is " << ComputeTotalResidualSynced(stream);
	
	//Solve it and update
	SolvePCGMaterialized();
	m_iteration_data.ApplyWarpFieldUpdate(stream);
}

void surfelwarp::WarpSolver::materializedFixedIndexSolverIterationSerial(cudaStream_t stream) {
	//Hand in the new SE3 to handlers
	m_dense_depth_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_density_foreground_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_sparse_correspondence_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	
	//The computation of jacobian
	ComputeTermJacobianFixedIndex(stream, stream, stream, stream);
	
	//The computation of diagonal blks JtJ and JtError
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditioner(stream);
	ComputeJtResidual(stream);
	
	//The non-diagonal blks of the matrix
	SetJtJMaterializerInput();
	MaterializeJtJNondiagonalBlocks(stream);
	
	//The assemble of matrix
	const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, stream);
	
	//Debug methods
	LOG(INFO) << "The total squared residual in materialized, fixed-index solver is " << ComputeTotalResidualSynced(stream);
	
	//Solve it and update
	SolvePCGMaterialized();
	m_iteration_data.ApplyWarpFieldUpdate(stream);
}


//The materialized, fixed-index solver with distinguished local/global iteration
void surfelwarp::WarpSolver::solveMaterializedFixedIndexGlobalLocalSerial(cudaStream_t stream) {
	QueryPixelKNN(stream);
	fullGlobalSolverIterationMaterializedFixedIndexSerial(stream);
	for(auto i = 0; i < Constants::kNumGaussNewtonIterations - 1; i++) {
		if(m_iteration_data.IsGlobalIteration())
			materializedFixedIndexSolverGlobalIterationSerial(stream);
		else
			materializedFixedIndexSolverIterationSerial(stream);
	}
}


void surfelwarp::WarpSolver::fullGlobalSolverIterationMaterializedFixedIndexSerial(cudaStream_t stream) {
	FetchPotentialDenseImageTermPixelsFixedIndexSynced(stream);
	setDenseDepthHandlerFullInput();
	setDensityForegroundHandlerFullInput();
	FindPotentialForegroundMaskPixelSynced(stream);
	SelectValidSparseFeatureMatchedPairs(stream);
	
	//The indexing part
	SetNode2TermIndexInput();
	BuildNode2TermIndex(stream);
	BuildNodePair2TermIndexBlocked(stream);
	
	//The computation of jacobian
	ComputeTermJacobianFixedIndex(stream, stream, stream, stream);
	
	//The computation of diagonal blks JtJ and JtError
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditionerGlobalIteration(stream);
	ComputeJtResidualGlobalIteration(stream);
	
	//The non-diagonal blks of the matrix
	SetJtJMaterializerInput();
	MaterializeJtJNondiagonalBlocksGlobalIteration(stream);
	
	//The assemble of matrix
	const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, stream);
	
	//Debug methods
	LOG(INFO) << "The total squared residual in materialized, fixed-index solver is " << ComputeTotalResidualSynced(stream);
	
	//Solve it and update
	SolvePCGMaterialized();
	m_iteration_data.ApplyWarpFieldUpdate(stream);
}

void surfelwarp::WarpSolver::materializedFixedIndexSolverGlobalIterationSerial(cudaStream_t stream) {
	//Hand in the new SE3 to handlers
	m_dense_depth_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_density_foreground_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_sparse_correspondence_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	
	//The computation of jacobian
	ComputeTermJacobianFixedIndex(stream, stream, stream, stream);
	
	//The computation of diagonal blks JtJ and JtError
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditionerGlobalIteration(stream);
	ComputeJtResidualGlobalIteration(stream);
	
	//The non-diagonal blks of the matrix
	SetJtJMaterializerInput();
	MaterializeJtJNondiagonalBlocksGlobalIteration(stream);
	
	//The assemble of matrix
	const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, stream);
	
	//Debug methods
	LOG(INFO) << "The total squared residual in materialized, fixed-index solver is " << ComputeTotalResidualSynced(stream);
	
	//Solve it and update
	SolvePCGMaterialized();
	m_iteration_data.ApplyWarpFieldUpdate(stream);
}