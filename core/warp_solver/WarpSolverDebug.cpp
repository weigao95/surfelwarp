#include "core/warp_solver/WarpSolver.h"
#include "pcg_solver/ApplySpMVBase.h"

void surfelwarp::WarpSolver::TestSolver() {
	QueryPixelKNN();
	FindCorrespondDepthPixelPairsFreeIndex();
	FindValidColorForegroundMaskPixel();
	SelectValidSparseFeatureMatchedPairs();
	SetNode2TermIndexInput();
	BuildNode2TermIndex();
#if defined(USE_MATERIALIZED_JTJ)
	BuildNodePair2TermIndexBlocked();
#endif
	ComputeTermJacobiansFreeIndex();
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditioner();
	ComputeJtResidual();
	
	//These are debug methods
	SetJtJMaterializerInput();
	MaterializeJtJNondiagonalBlocks();
	
	//The diagonal blks
	auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks);
	
	//The debug method
	LOG(INFO) << "Prepare the data to check materialized spmv";
	std::vector<float> x_h; x_h.resize(m_apply_jtj_handler->MatrixSize());
	fillRandomVector(x_h);
	DeviceArray<float> x_dev, spmv_x;
	spmv_x.create(x_h.size());
	x_dev.upload(x_h);
	m_apply_jtj_handler->ApplyJtJ(DeviceArrayView<float>(x_dev), DeviceArraySlice<float>(spmv_x));
	
	//Now check it
	m_jtj_materializer->TestSparseMV(DeviceArrayView<float>(x_dev), DeviceArrayView<float>(spmv_x));
}

void surfelwarp::WarpSolver::FullSolverIterationTest() {
	QueryPixelKNN();
	FindCorrespondDepthPixelPairsFreeIndex();
	FindValidColorForegroundMaskPixel();
	SelectValidSparseFeatureMatchedPairs();
	SetNode2TermIndexInput();
	BuildNode2TermIndex();
#if defined(USE_MATERIALIZED_JTJ)
	BuildNodePair2TermIndexBlocked();
#endif
	ComputeTermJacobiansFreeIndex();
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditioner();
	ComputeJtResidual();
	
	//These are debug methods
	//SetJtJMaterializerInput();
	//MaterializeJtJNondiagonalBlocks();
	
	//The diagonal blks
	//auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	//m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks);
	
	//These are debug methods
	LOG(INFO) << "The total squared residual is " << ComputeTotalResidualSynced();
	
	//Solve it and update
	SolvePCGMatrixFree();
	//SolvePCGMaterialized();
	m_iteration_data.ApplyWarpFieldUpdate();
}


void surfelwarp::WarpSolver::MaterializedFullSolverIterationTest() {
	QueryPixelKNN();
	FindCorrespondDepthPixelPairsFreeIndex();
	FindValidColorForegroundMaskPixel();
	SelectValidSparseFeatureMatchedPairs();
	SetNode2TermIndexInput();
	BuildNode2TermIndex();
	BuildNodePair2TermIndexBlocked();
	ComputeTermJacobiansFreeIndex();
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditioner();
	ComputeJtResidual();
	
	//The materialization of the matrix
	SetJtJMaterializerInput();
	MaterializeJtJNondiagonalBlocks();
	
	//The diagonal blks
	auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks);
	
	LOG(INFO) << "The total squared residual is " << ComputeTotalResidualSynced();
	
	//Solve it and update
	auto matrixfree_spmv_handler = m_apply_jtj_handler;
	auto materialized_spmv_handler = m_jtj_materializer->GetSpMVHandler();
	//ApplySpMVBase<6>::CompareApplySpMV(matrixfree_spmv_handler, materialized_spmv_handler);
	
	//SolvePCGMatrixFree();
	SolvePCGMaterialized();
	m_iteration_data.ApplyWarpFieldUpdate();
	
	//Test the reduce
	FindCorrespondDepthPixelPairsFreeIndex();
	FindValidColorForegroundMaskPixel();
	SelectValidSparseFeatureMatchedPairs();
	SetNode2TermIndexInput();
	BuildNode2TermIndex();
	BuildNodePair2TermIndexBlocked();
	ComputeTermJacobiansFreeIndex();
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditioner();
	ComputeJtResidual();
	
	
	LOG(INFO) << "The total squared residual is " << ComputeTotalResidualSynced();
}


void surfelwarp::WarpSolver::SolverIterationWithIndexTest() {
	LOG(FATAL) << "Outdated";
	m_dense_depth_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_density_foreground_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_sparse_correspondence_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	ComputeTermJacobiansFreeIndex();
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditioner();
	ComputeJtResidual();
	
	//These are debug methods
	LOG(INFO) << "The total squared residual is " << ComputeTotalResidualSynced();
	
	//Solve it and update
	SolvePCGMatrixFree();
	m_iteration_data.ApplyWarpFieldUpdate();
}


/* These are some failed/deprecated exploration.
 */

//The interface for matrix-free index-free solver
void surfelwarp::WarpSolver::solveMatrixIndexFreeSerial(cudaStream_t stream) {
	QueryPixelKNN(stream);
	for (auto i = 0; i < Constants::kNumGaussNewtonIterations; i++) {
		fullSolverIterationMatrixFreeSerial(stream);
	}
}


void surfelwarp::WarpSolver::fullSolverIterationMatrixFreeSerial(cudaStream_t stream) {
	FindCorrespondDepthPixelPairsFreeIndex(stream);
	FindValidColorForegroundMaskPixel(stream, stream);
	SelectValidSparseFeatureMatchedPairs(stream);
	SetNode2TermIndexInput();
	BuildNode2TermIndex(stream);
	ComputeTermJacobiansFreeIndex(stream, stream, stream, stream);
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditioner(stream);
	ComputeJtResidual(stream);

	//Debug methods
	LOG(INFO) << "The total squared residual is " << ComputeTotalResidualSynced(stream);

	//Solve it and update
	SolvePCGMatrixFree();
	m_iteration_data.ApplyWarpFieldUpdate(stream);
}

//The solver interface for materialized, index free solver
void surfelwarp::WarpSolver::solveMaterializedIndexFreeSerial(cudaStream_t stream) {
	QueryPixelKNN(stream);
	for (auto i = 0; i < Constants::kNumGaussNewtonIterations; i++) {
		fullSolverIterationMaterializedIndexFreeSerial(stream);
	}
}

void surfelwarp::WarpSolver::fullSolverIterationMaterializedIndexFreeSerial(cudaStream_t stream) {
	FindCorrespondDepthPixelPairsFreeIndex(stream);
	FindValidColorForegroundMaskPixel(stream, stream);
	SelectValidSparseFeatureMatchedPairs(stream);
	SetNode2TermIndexInput();
	BuildNode2TermIndex(stream);
	BuildNodePair2TermIndexBlocked(stream);
	ComputeTermJacobiansFreeIndex(stream, stream, stream, stream);
	SetPreconditionerBuilderAndJtJApplierInput();
	BuildPreconditioner(stream);
	ComputeJtResidual(stream);

	//The materialization of the matrix
	SetJtJMaterializerInput();
	MaterializeJtJNondiagonalBlocks(stream);

	//The diagonal blks
	const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, stream);

	LOG(INFO) << "The total squared residual is " << ComputeTotalResidualSynced(stream);

	//Solve and update
	SolvePCGMaterialized();
	m_iteration_data.ApplyWarpFieldUpdate(stream);
}


//The materialized solver with lazy evaluation
void surfelwarp::WarpSolver::solveMaterializedLazyEvaluateSerial(cudaStream_t stream) {
	QueryPixelKNN(stream);
	fullSolverIterationMaterializedLazyEvaluateSerial(stream);
	for (auto i = 0; i < Constants::kNumGaussNewtonIterations; i++) {
		materializedLazyEvaluateSolverIterationSerial(stream);
	}
}

void surfelwarp::WarpSolver::fullSolverIterationMaterializedLazyEvaluateSerial(cudaStream_t stream) {
	//The first iteration, always construct the matrix
	fullSolverIterationMaterializedFixedIndexSerial(stream);
}

void surfelwarp::WarpSolver::materializedLazyEvaluateSolverIterationSerial(cudaStream_t stream) {
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

	//Should recompute the matrix?
	if (m_iteration_data.ComputeJtJLazyEvaluation()) {
		//The non-diagonal blks of the matrix
		SetJtJMaterializerInput();
		MaterializeJtJNondiagonalBlocks(stream);

		//The assemble of matrix
		const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
		m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, stream);
	}

	//Debug methods
	LOG(INFO) << "The total squared residual in materialized, fixed-index, lazy-evaluated solver is " << ComputeTotalResidualSynced(stream);

	//Solve it and update
	SolvePCGMaterialized(15);
	m_iteration_data.ApplyWarpFieldUpdate(stream);
}