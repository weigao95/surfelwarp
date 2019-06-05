//
// Created by wei on 3/28/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/CameraObservation.h"
#include "common/ArraySlice.h"
#include "common/surfel_types.h"
#include "core/render/Renderer.h"
#include "core/SurfelGeometry.h"
#include "core/WarpField.h"
#include "core/warp_solver/SolverIterationData.h"
#include "core/warp_solver/ImageTermKNNFetcher.h"
#include "core/warp_solver/DenseDepthHandler.h"
#include "core/warp_solver/DensityForegroundMapHandler.h"
#include "core/warp_solver/SparseCorrespondenceHandler.h"
#include "core/warp_solver/Node2TermsIndex.h"
#include "core/warp_solver/NodePair2TermsIndex.h"
#include "core/warp_solver/PreconditionerRhsBuilder.h"
#include "core/warp_solver/ResidualEvaluator.h"
#include "core/warp_solver/JtJMaterializer.h"
#include "core/warp_solver/ApplyJtJMatrixFreeHandler.h"
#include "core/warp_solver/NodeGraphSmoothHandler.h"
#include "pcg_solver/BlockPCG.h"
#include "math/DualQuaternion.hpp"
#include <memory>

namespace surfelwarp {
	
	class WarpSolver {
	private:
		//Default parameters
		int m_image_width;
		int m_image_height;
		
		//The inputs to the solver
		CameraObservation m_observation;
		Renderer::SolverMaps m_rendered_maps;
		SurfelGeometry::SolverInput m_geometry_input;
		WarpField::SolverInput m_warpfield_input;
		mat34 m_world2camera;
		
		//The interation data maintained by the solver
		SolverIterationData m_iteration_data;
		
	public:
		using Ptr = std::shared_ptr<WarpSolver>;
		WarpSolver();
		~WarpSolver() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(WarpSolver);
		
		//Matrix-free solver might override these methods;
		void AllocateBuffer();
		void ReleaseBuffer();
		
		
		//The maps and arrays accessed by solver
		void SetSolverInputs(
			CameraObservation observation,
			Renderer::SolverMaps rendered_maps,
			SurfelGeometry::SolverInput geometry_input,
			WarpField::SolverInput warpfield_input,
			const Matrix4f& world2camera
		);
		void SetSolverInputs(
			CameraObservation observation,
			Renderer::SolverMaps rendered_maps,
			SurfelGeometry::SolverInput geometry_input,
			WarpField::SolverInput warpfield_input,
			const mat34& world2camera
		);
		
		//The access interface
		DeviceArrayView<DualQuaternion> SolvedNodeSE3() const { return m_iteration_data.CurrentWarpFieldInput(); }
		
		//The serial solver interface
		void SolveSerial(cudaStream_t stream = 0);
		
		
		
		//These are debug interface
	public:
		void TestSolver();
		void FullSolverIterationTest();
		void MaterializedFullSolverIterationTest();
		void SolverIterationWithIndexTest();
		
		
		/* Query the KNN for pixels given index map
		 * The knn map is in the same resolution as image
		 */
	private:
		DeviceArray2D<KNNAndWeight> m_knn_map;
	public:
		void QueryPixelKNN(cudaStream_t stream = 0);
		
		
		/* Fetch the potential valid image term pixels, knn and weight
		 */
	private:
		ImageTermKNNFetcher::Ptr m_image_knn_fetcher;
		void allocateImageKNNFetcherBuffer();
		void releaseImageKNNFetcherBuffer();
	public:
		void FetchPotentialDenseImageTermPixelsFixedIndexSynced(cudaStream_t stream = 0);
		
		
		/* Hand in the geometry maps to
		 * depth correspondence finder
		 * Depends: QueryPixelKNN, FetchPotentialDenseImageTermPixels
		 */
	private:
		DenseDepthHandler::Ptr m_dense_depth_handler;
		void allocateDenseDepthBuffer();
		void releaseDenseDepthBuffer();
		void setDenseDepthHandlerFullInput();
	public:
		//The method for indexing
		void FindCorrespondDepthPixelPairsFreeIndex(cudaStream_t stream = 0);
		
		//The method to compute alignment error after solving
		void ComputeAlignmentErrorMapDirect(cudaStream_t stream = 0);
		void ComputeAlignmentErrorOnNodes(cudaStream_t stream = 0);
		void ComputeAlignmentErrorMapFromNode(cudaStream_t stream = 0);
		cudaTextureObject_t GetAlignmentErrorMap() const { return m_dense_depth_handler->GetAlignmentErrorMap(); }
		NodeAlignmentError GetNodeAlignmentError() const { return m_dense_depth_handler->GetNodeAlignmentError(); }

		
		/* Hand in the color and foreground
		 * mask to valid pixel compactor
		 * Depends: QueryPixelKNN
		 */
	private:
		DensityForegroundMapHandler::Ptr m_density_foreground_handler;
		void allocateDensityForegroundMapBuffer();
		void releaseDensityForegroundMapBuffer();
		void setDensityForegroundHandlerFullInput();
	public:
		void FindValidColorForegroundMaskPixel(cudaStream_t color_stream = 0, cudaStream_t mask_stream = 0);
		void FindPotentialForegroundMaskPixelSynced(cudaStream_t stream = 0);
		
		
		/* Hand in the vertex maps and pixel pairs
		 * to sparse feature handler
		 */
	private:
		SparseCorrespondenceHandler::Ptr m_sparse_correspondence_handler;
		void allocateSparseFeatureBuffer();
		void releaseSparseFeatureBuffer();
	public:
		void SetSparseFeatureHandlerFullInput();
		void SelectValidSparseFeatureMatchedPairs(cudaStream_t stream = 0);
		
		
		/* Hand in the value to node graph term handler
		 */
	private:
		NodeGraphSmoothHandler::Ptr m_graph_smooth_handler;
		void allocateSmoothTermHandlerBuffer();
		void releaseSmoothTermHandlerBuffer();
		void computeSmoothTermNode2Jacobian(cudaStream_t stream);
		
		
		/* Build the node to term index
		 * Depends: correspond depth, valid pixel, node graph, sparse feature
		 */
	private:
		Node2TermsIndex::Ptr m_node2term_index;
		NodePair2TermsIndex::Ptr m_nodepair2term_index;
		void allocateNode2TermIndexBuffer();
		void releaseNode2TermIndexBuffer();
	public:
		void SetNode2TermIndexInput();
		void BuildNode2TermIndex(cudaStream_t stream = 0);
		void BuildNodePair2TermIndexBlocked(cudaStream_t stream = 0);
		
		
		/* Compute the jacobians for all terms
		 */
	public:
		void ComputeTermJacobiansFreeIndex(
			cudaStream_t dense_depth = 0,
			cudaStream_t density_map = 0,
			cudaStream_t foreground_mask = 0,
			cudaStream_t sparse_feature = 0
		);
		void ComputeTermJacobianFixedIndex(
			cudaStream_t dense_depth = 0,
			cudaStream_t density_map = 0,
			cudaStream_t foreground_mask = 0,
			cudaStream_t sparse_feature = 0
		);
		
		
		
		/* Construct the preconditioner and rhs of the method
		 */
	private:
		PreconditionerRhsBuilder::Ptr m_preconditioner_rhs_builder;
		void allocatePreconditionerRhsBuffer();
		void releasePreconditionerRhsBuffer();
	public:
		void SetPreconditionerBuilderAndJtJApplierInput();
		
		//The interface for diagonal pre-conditioner
		void BuildPreconditioner(cudaStream_t stream = 0);
		void BuildPreconditionerGlobalIteration(cudaStream_t stream = 0);
		
		//The interface for jt residual
		void ComputeJtResidual(cudaStream_t stream = 0);
		void ComputeJtResidualGlobalIteration(cudaStream_t stream = 0);
		DeviceArrayView<float> JtResidualValue() const { return m_preconditioner_rhs_builder->JtDotResidualValue(); }
		DeviceArrayView<float> JtJDiagonalBlockValue() const { return m_preconditioner_rhs_builder->JtJDiagonalBlocks(); }
		
		
		/* The residual evaluator, the input is the same as above
		 */
	private:
		ResidualEvaluator::Ptr m_residual_evaluator;
		void allocateResidualEvaluatorBuffer();
		void releaseResidualEvaluatorBuffer();
	public:
		float ComputeTotalResidualSynced(cudaStream_t stream = 0);
		
		
		/* Materialize the JtJ matrix
		 */
	private:
		JtJMaterializer::Ptr m_jtj_materializer;
		void allocateMaterializedJtJBuffer();
		void releaseMaterializedJtJBuffer();
	public:
		void SetJtJMaterializerInput();
		void MaterializeJtJNondiagonalBlocks(cudaStream_t stream = 0);
		void MaterializeJtJNondiagonalBlocksGlobalIteration(cudaStream_t stream = 0);


		/* The method to apply JtJ to a vector
		 */
	private:
		ApplyJtJHandlerMatrixFree::Ptr m_apply_jtj_handler;
		void allocateJtJApplyBuffer();
		void releaseJtJApplyBuffer();
		
		
		/* The pcg solver
		 */
	private:
		BlockPCG<6>::Ptr m_pcg_solver;
		void allocatePCGSolverBuffer();
		void releasePCGSolverBuffer();
	public:
		void UpdatePCGSolverStream(cudaStream_t stream);
		void SolvePCGMatrixFree();
		void SolvePCGMaterialized(int pcg_iterations = 10);


		/* The solver interface for streamed solver. This solver needs
		 * to break the encapsulation of functions for best performance
		 * See core/warp_solver/WarpSolverStreamed.cpp for details
		 */
	private:
		cudaStream_t m_solver_stream[4];
		void initSolverStream();
		void releaseSolverStream();
		void syncAllSolverStream();

		void buildSolverIndexStreamed();
		void solverIterationGlobalIterationStreamed();
		void solverIterationLocalIterationStreamed();
	public:
		void SolveStreamed();


		//These are private interface
	private:
		//The matrix-free index free solver interface
		void solveMatrixIndexFreeSerial(cudaStream_t stream = 0);
		void fullSolverIterationMatrixFreeSerial(cudaStream_t stream = 0);

		//The matrix-free solver which build index in the first iteration, and reuse the index
		void solveMatrixFreeFixedIndexSerial(cudaStream_t stream = 0);
		void fullSolverIterationMatrixFreeFixedIndexSerial(cudaStream_t stream = 0);
		void matrixFreeFixedIndexSolverIterationSerial(cudaStream_t stream = 0);

		//The materialized index-free solver interface
		void solveMaterializedIndexFreeSerial(cudaStream_t stream = 0);
		void fullSolverIterationMaterializedIndexFreeSerial(cudaStream_t stream = 0);

		//The materialized fixed index solver interface
		void solveMaterializedFixedIndexSerial(cudaStream_t stream = 0);
		void fullSolverIterationMaterializedFixedIndexSerial(cudaStream_t stream = 0);
		void materializedFixedIndexSolverIterationSerial(cudaStream_t stream = 0);

		//The one distinguish between local and global iteration
		void solveMaterializedFixedIndexGlobalLocalSerial(cudaStream_t stream = 0);
		void fullGlobalSolverIterationMaterializedFixedIndexSerial(cudaStream_t stream = 0);
		void materializedFixedIndexSolverGlobalIterationSerial(cudaStream_t stream = 0);


		//The materialized, fixed-index and lazy evaluation solver interface
		void solveMaterializedLazyEvaluateSerial(cudaStream_t stream = 0);
		void fullSolverIterationMaterializedLazyEvaluateSerial(cudaStream_t stream = 0);
		void materializedLazyEvaluateSolverIterationSerial(cudaStream_t stream = 0);
	};
	
	
}
