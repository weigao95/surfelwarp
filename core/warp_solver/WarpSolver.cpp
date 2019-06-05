//
// Created by wei on 3/28/18.
//

#include <core/WarpField.h>
#include "common/ConfigParser.h"
#include "common/CameraObservation.h"
#include "core/warp_solver/WarpSolver.h"
#include "core/warp_solver/solver_constants.h"


/* The constructor only zero-init
 * The caller ensures allocate/release
 */
surfelwarp::WarpSolver::WarpSolver() : m_iteration_data() {
	//Query the image
	const auto config = ConfigParser::Instance();
	m_image_height = config.clip_image_rows();
	m_image_width = config.clip_image_cols();
	
	memset(&m_observation, 0, sizeof(m_observation));
	memset(&m_rendered_maps, 0, sizeof(m_rendered_maps));
	memset(&m_geometry_input, 0, sizeof(m_geometry_input));
}

void surfelwarp::WarpSolver::AllocateBuffer() {
	m_knn_map.create(m_image_height, m_image_width);
	
	//The correspondence depth and albedo pixel pairs
	allocateImageKNNFetcherBuffer();
	allocateDenseDepthBuffer();
	allocateDensityForegroundMapBuffer();
	allocateSparseFeatureBuffer();
	allocateSmoothTermHandlerBuffer();
	allocateNode2TermIndexBuffer();
	allocatePreconditionerRhsBuffer();
	allocateResidualEvaluatorBuffer();
	allocateJtJApplyBuffer();
#if defined(USE_MATERIALIZED_JTJ)
	allocateMaterializedJtJBuffer();
#endif
	allocatePCGSolverBuffer();
	
	//Init the stream for cuda
	initSolverStream();
}

void surfelwarp::WarpSolver::ReleaseBuffer() {
	m_knn_map.release();

	//Destroy the stream
	releaseSolverStream();
	
	//Release the corresponded buffer
	releaseImageKNNFetcherBuffer();
	releaseDenseDepthBuffer();
	releaseDensityForegroundMapBuffer();
	releaseSparseFeatureBuffer();
	releaseSmoothTermHandlerBuffer();
	releaseNode2TermIndexBuffer();
	releasePreconditionerRhsBuffer();
	releaseResidualEvaluatorBuffer();
	releaseJtJApplyBuffer();
#if defined(USE_MATERIALIZED_JTJ)
	releaseMaterializedJtJBuffer();
#endif
	releasePCGSolverBuffer();
}


/* The buffer and method for dense image term
 */
void surfelwarp::WarpSolver::allocateImageKNNFetcherBuffer() {
	m_image_knn_fetcher = std::make_shared<ImageTermKNNFetcher>();
}

void surfelwarp::WarpSolver::releaseImageKNNFetcherBuffer() {
}

void surfelwarp::WarpSolver::FetchPotentialDenseImageTermPixelsFixedIndexSynced(cudaStream_t stream) {
	//Hand in the input
	m_image_knn_fetcher->SetInputs(m_knn_map, m_rendered_maps.index_map);
	
	//Do processing
	m_image_knn_fetcher->MarkPotentialMatchedPixels(stream);
	m_image_knn_fetcher->CompactPotentialValidPixels(stream);
	m_image_knn_fetcher->SyncQueryCompactedPotentialPixelSize(stream);
	
	//The sanity check: seems correct
	//Call this after dense depth handler
	//const auto& dense_depth_knn = m_dense_depth_handler->DenseDepthTermsKNNArray();
	//m_image_knn_fetcher->CheckDenseImageTermKNN(dense_depth_knn);
}

/* The buffer and method for correspondence finder
 */
void surfelwarp::WarpSolver::allocateDenseDepthBuffer() {
	m_dense_depth_handler = std::make_shared<DenseDepthHandler>();
	m_dense_depth_handler->AllocateBuffer();
}

void surfelwarp::WarpSolver::releaseDenseDepthBuffer() {
	m_dense_depth_handler->ReleaseBuffer();
}

void surfelwarp::WarpSolver::setDenseDepthHandlerFullInput() {
	const auto node_se3 = m_iteration_data.CurrentWarpFieldInput();
	
	//Construct the input
	m_dense_depth_handler->SetInputs(
		node_se3,
		m_knn_map,
		m_observation.vertex_config_map,
		m_observation.normal_radius_map,
		m_rendered_maps.reference_vertex_map,
		m_rendered_maps.reference_normal_map,
		m_rendered_maps.index_map,
		m_world2camera,
		m_image_knn_fetcher->GetImageTermPixelAndKNN()
	);
}

void surfelwarp::WarpSolver::FindCorrespondDepthPixelPairsFreeIndex(cudaStream_t stream) {
	setDenseDepthHandlerFullInput();
	m_dense_depth_handler->FindCorrespondenceSynced(stream);
}

void surfelwarp::WarpSolver::ComputeAlignmentErrorMapDirect(cudaStream_t stream) {
	const auto node_se3 = m_iteration_data.CurrentWarpFieldInput();
	m_dense_depth_handler->ComputeAlignmentErrorMapDirect(
		node_se3,
		m_world2camera,
		m_observation.filter_foreground_mask,
		stream
	);
}


void surfelwarp::WarpSolver::ComputeAlignmentErrorOnNodes(cudaStream_t stream) {
	const auto node_se3 = m_iteration_data.CurrentWarpFieldInput();
	m_dense_depth_handler->ComputeNodewiseError(
		node_se3,
		m_world2camera,
		m_observation.filter_foreground_mask,
		stream
	);
}


void surfelwarp::WarpSolver::ComputeAlignmentErrorMapFromNode(cudaStream_t stream) {
	const auto node_se3 = m_iteration_data.CurrentWarpFieldInput();
	m_dense_depth_handler->ComputeAlignmentErrorMapFromNode(
		node_se3, 
		m_world2camera,
		m_observation.filter_foreground_mask,
		stream
	);
}


/* The buffer and method for density and foreground mask pixel finder
 */
void surfelwarp::WarpSolver::allocateDensityForegroundMapBuffer() {
	m_density_foreground_handler = std::make_shared<DensityForegroundMapHandler>();
	m_density_foreground_handler->AllocateBuffer();
}

void surfelwarp::WarpSolver::releaseDensityForegroundMapBuffer() {
	m_density_foreground_handler->ReleaseBuffer();
}

void surfelwarp::WarpSolver::setDensityForegroundHandlerFullInput() {
	//The current node se3 from iteraion data
	const auto node_se3 = m_iteration_data.CurrentWarpFieldInput();

	//Hand in the information to handler
#if defined(USE_RENDERED_RGBA_MAP_SOLVER)
	m_density_foreground_handler->SetInputs(
		node_se3,
		m_knn_map,
		m_observation.foreground_mask,
		m_observation.filter_foreground_mask,
		m_observation.foreground_mask_gradient_map,
		m_observation.density_map,
		m_observation.density_gradient_map,
		m_rendered_maps.reference_vertex_map,
		m_rendered_maps.reference_normal_map,
		m_rendered_maps.index_map,
		m_rendered_maps.normalized_rgb_map,
		m_world2camera,
		m_image_knn_fetcher->GetImageTermPixelAndKNN()
	);
#else
	m_density_foreground_handler->SetInputs(
		node_se3,
		m_knn_map,
		m_observation.foreground_mask,
		m_observation.filter_foreground_mask,
		m_observation.foreground_mask_gradient_map,
		m_observation.density_map,
		m_observation.density_gradient_map,
		m_rendered_maps.reference_vertex_map,
		m_rendered_maps.reference_normal_map,
		m_rendered_maps.index_map,
		m_observation.normalized_rgba_prevframe,
		m_world2camera,
		m_image_knn_fetcher->GetImageTermPixelAndKNN()
	);
#endif
}

void surfelwarp::WarpSolver::FindValidColorForegroundMaskPixel(cudaStream_t color_stream, cudaStream_t mask_stream)
{
	//Provide to input
	setDensityForegroundHandlerFullInput();
	
	//Do it
	m_density_foreground_handler->FindValidColorForegroundMaskPixels(color_stream, mask_stream);
}

void surfelwarp::WarpSolver::FindPotentialForegroundMaskPixelSynced(cudaStream_t stream) {
	//Provide to input
	setDensityForegroundHandlerFullInput();

	//Do it
	m_density_foreground_handler->FindPotentialForegroundMaskPixelSynced(stream);
}


/* The method to filter the sparse feature term
 */
void surfelwarp::WarpSolver::allocateSparseFeatureBuffer() {
	m_sparse_correspondence_handler = std::make_shared<SparseCorrespondenceHandler>();
	m_sparse_correspondence_handler->AllocateBuffer();
}

void surfelwarp::WarpSolver::releaseSparseFeatureBuffer() {
	m_sparse_correspondence_handler->ReleaseBuffer();
}

void surfelwarp::WarpSolver::SetSparseFeatureHandlerFullInput() {
	//The current node se3 from iteraion data
	const auto node_se3 = m_iteration_data.CurrentWarpFieldInput();

	m_sparse_correspondence_handler->SetInputs(
		node_se3,
		m_knn_map,
		m_observation.vertex_config_map,
		m_observation.correspondence_pixel_pairs,
		m_rendered_maps.reference_vertex_map,
		m_rendered_maps.index_map,
		m_world2camera
	);
}

void surfelwarp::WarpSolver::SelectValidSparseFeatureMatchedPairs(cudaStream_t stream) {
	SetSparseFeatureHandlerFullInput();
	m_sparse_correspondence_handler->BuildCorrespondVertexKNN(stream);
}

/* The method for smooth term handler
 */
void surfelwarp::WarpSolver::allocateSmoothTermHandlerBuffer() {
	m_graph_smooth_handler = std::make_shared<NodeGraphSmoothHandler>();
}

void surfelwarp::WarpSolver::releaseSmoothTermHandlerBuffer() {

}

void surfelwarp::WarpSolver::computeSmoothTermNode2Jacobian(cudaStream_t stream) {
	//Prepare the input
	m_graph_smooth_handler->SetInputs(
		m_iteration_data.CurrentWarpFieldInput(),
		m_warpfield_input.node_graph,
		m_warpfield_input.reference_node_coords
	);
	
	//Do it
	m_graph_smooth_handler->BuildTerm2Jacobian(stream);
}

/* The index from node to term index
 */
void surfelwarp::WarpSolver::allocateNode2TermIndexBuffer() {
	m_node2term_index = std::make_shared<Node2TermsIndex>();
	m_node2term_index->AllocateBuffer();
#if defined(USE_MATERIALIZED_JTJ)
	m_nodepair2term_index = std::make_shared<NodePair2TermsIndex>();
	m_nodepair2term_index->AllocateBuffer();
#endif
}

void surfelwarp::WarpSolver::releaseNode2TermIndexBuffer() {
	m_node2term_index->ReleaseBuffer();
#if defined(USE_MATERIALIZED_JTJ)
	m_nodepair2term_index->ReleaseBuffer();
#endif
}

void surfelwarp::WarpSolver::SetNode2TermIndexInput() {
	const auto dense_depth_knn = m_image_knn_fetcher->DenseImageTermKNNArray();
	//const auto density_map_knn = m_image_knn_fetcher->DenseImageTermKNNArray();
	const auto density_map_knn = DeviceArrayView<ushort4>(); //Empty
	const auto node_graph = m_warpfield_input.node_graph;
	const auto foreground_mask_knn = m_density_foreground_handler->ForegroundMaskTermKNN();
	const auto sparse_feature_knn = m_sparse_correspondence_handler->SparseFeatureKNN();
	m_node2term_index->SetInputs(
		dense_depth_knn,
		node_graph,
		m_warpfield_input.node_se3.Size(),
		foreground_mask_knn,
		sparse_feature_knn
	);

#if defined(USE_MATERIALIZED_JTJ)
	const auto num_nodes = m_warpfield_input.node_se3.Size();
	m_nodepair2term_index->SetInputs(
		num_nodes,
		dense_depth_knn,
		node_graph,
		foreground_mask_knn,
		sparse_feature_knn
	);
#endif
}

void surfelwarp::WarpSolver::BuildNode2TermIndex(cudaStream_t stream) {
	m_node2term_index->BuildIndex(stream);
}

void surfelwarp::WarpSolver::BuildNodePair2TermIndexBlocked(cudaStream_t stream) {
	m_nodepair2term_index->BuildHalfIndex(stream);
	m_nodepair2term_index->QueryValidNodePairSize(stream); //This will blocked
	
	//The later computation depends on the size
	m_nodepair2term_index->BuildSymmetricAndRowBlocksIndex(stream);
	
	
	//Do a sanity check
	//m_nodepair2term_index->CheckHalfIndex();
	//m_nodepair2term_index->CompactedIndexSanityCheck();
	//m_nodepair2term_index->IndexStatistics();
	
	//Do a statistic on the smooth term: cannot do this
	//The non-image terms are curcial to solver stability
	//m_nodepair2term_index->CheckSmoothTermIndexCompleteness();
}

/* Prepare the jacobian for later use
 */
void surfelwarp::WarpSolver::ComputeTermJacobiansFreeIndex(
	cudaStream_t dense_depth, cudaStream_t density_map,
	cudaStream_t foreground_mask, cudaStream_t sparse_feature
) {
	m_dense_depth_handler->ComputeJacobianTermsFreeIndex(dense_depth);
	computeSmoothTermNode2Jacobian(sparse_feature);
	m_density_foreground_handler->ComputeTwistGradient(density_map, foreground_mask);
	m_sparse_correspondence_handler->BuildTerm2Jacobian(sparse_feature);
}

//Assume the SE3 for each term expepted smooth term is updated
void surfelwarp::WarpSolver::ComputeTermJacobianFixedIndex(
	cudaStream_t dense_depth,
	cudaStream_t density_map,
	cudaStream_t foreground_mask,
	cudaStream_t sparse_feature
) {
	m_dense_depth_handler->ComputeJacobianTermsFixedIndex(dense_depth);
	computeSmoothTermNode2Jacobian(sparse_feature);
	m_density_foreground_handler->ComputeTwistGradient(density_map, foreground_mask);
	m_sparse_correspondence_handler->BuildTerm2Jacobian(sparse_feature);
}

/* Compute the preconditioner and linear equation rhs for later use
 */
void surfelwarp::WarpSolver::allocatePreconditionerRhsBuffer() {
	m_preconditioner_rhs_builder = std::make_shared<PreconditionerRhsBuilder>();
	m_preconditioner_rhs_builder->AllocateBuffer();
}

void surfelwarp::WarpSolver::releasePreconditionerRhsBuffer() {
	m_preconditioner_rhs_builder->ReleaseBuffer();
}

void surfelwarp::WarpSolver::SetPreconditionerBuilderAndJtJApplierInput() {
	//Map from node to term
	const auto node2term = m_node2term_index->GetNode2TermMap();
	
	//The dense depth term
	const auto dense_depth_term2jacobian = m_dense_depth_handler->Term2JacobianMap();
	
	//The node graph term
	const auto smooth_term2jacobian = m_graph_smooth_handler->Term2JacobianMap();
	
	//The image map term
	DensityMapTerm2Jacobian density_term2jacobian;
	ForegroundMaskTerm2Jacobian foreground_term2jacobian;
	m_density_foreground_handler->Term2JacobianMaps(density_term2jacobian, foreground_term2jacobian);
	
	//The sparse feature term
	const auto feature_term2jacobian = m_sparse_correspondence_handler->Term2JacobianMap();
	
	//The penalty constants
	const auto penalty_constants = m_iteration_data.CurrentPenaltyConstants();
	
	//Hand in the input to preconditioner builder
	m_preconditioner_rhs_builder->SetInputs(
		node2term,
		dense_depth_term2jacobian,
		smooth_term2jacobian,
		density_term2jacobian,
		foreground_term2jacobian,
		feature_term2jacobian,
		penalty_constants
	);
	
	//Hand in to residual evaluator
	m_residual_evaluator->SetInputs(
		node2term,
		dense_depth_term2jacobian,
		smooth_term2jacobian,
		density_term2jacobian,
		foreground_term2jacobian,
		feature_term2jacobian,
		penalty_constants
	);

	//Hand in the input to jtj applier
	m_apply_jtj_handler->SetInputs(
		node2term,
		dense_depth_term2jacobian,
		smooth_term2jacobian,
		density_term2jacobian,
		foreground_term2jacobian,
		feature_term2jacobian,
		penalty_constants
	);
}

void surfelwarp::WarpSolver::BuildPreconditioner(cudaStream_t stream) {
	m_preconditioner_rhs_builder->ComputeDiagonalPreconditioner(stream);
}

void surfelwarp::WarpSolver::BuildPreconditionerGlobalIteration(cudaStream_t stream) {
	m_preconditioner_rhs_builder->ComputeDiagonalPreconditionerGlobalIteration(stream);
}

//The method to compute jt residual
void surfelwarp::WarpSolver::ComputeJtResidual(cudaStream_t stream) {
	m_preconditioner_rhs_builder->ComputeJtResidual(stream);
}

void surfelwarp::WarpSolver::ComputeJtResidualGlobalIteration(cudaStream_t stream) {
	m_preconditioner_rhs_builder->ComputeJtResidualGlobalIteration(stream);
}



/* The method to materialized the JtJ matrix
 */
void surfelwarp::WarpSolver::allocateMaterializedJtJBuffer() {
	m_jtj_materializer = std::make_shared<JtJMaterializer>();
	m_jtj_materializer->AllocateBuffer();
}

void surfelwarp::WarpSolver::releaseMaterializedJtJBuffer() {
	m_jtj_materializer->ReleaseBuffer();
}

void surfelwarp::WarpSolver::SetJtJMaterializerInput() {
	//Map from node to term
	const auto node2term = m_node2term_index->GetNode2TermMap();
	
	//Map from nodepair to term
	const auto nodepair2term = m_nodepair2term_index->GetNodePair2TermMap();
	
	//The dense depth term
	const auto dense_depth_term2jacobian = m_dense_depth_handler->Term2JacobianMap();
	
	//The node graph term
	const auto smooth_term2jacobian = m_graph_smooth_handler->Term2JacobianMap();
	
	//The image map term
	DensityMapTerm2Jacobian density_term2jacobian;
	ForegroundMaskTerm2Jacobian foreground_term2jacobian;
	m_density_foreground_handler->Term2JacobianMaps(density_term2jacobian, foreground_term2jacobian);
	
	//The sparse feature term
	const auto feature_term2jacobian = m_sparse_correspondence_handler->Term2JacobianMap();
	
	//The penalty constants
	const auto penalty_constants = m_iteration_data.CurrentPenaltyConstants();
	
	//Hand in to materializer
	m_jtj_materializer->SetInputs(
		nodepair2term,
		dense_depth_term2jacobian,
		smooth_term2jacobian,
		density_term2jacobian,
		foreground_term2jacobian,
		feature_term2jacobian,
		node2term,
		penalty_constants
	);
}

void surfelwarp::WarpSolver::MaterializeJtJNondiagonalBlocks(cudaStream_t stream) {
	m_jtj_materializer->BuildMaterializedJtJNondiagonalBlocks(stream);
}

void surfelwarp::WarpSolver::MaterializeJtJNondiagonalBlocksGlobalIteration(cudaStream_t stream) {
	m_jtj_materializer->BuildMaterializedJtJNondiagonalBlocksGlobalIteration(stream);
}


/* The method to apply JtJ to a given vector
 */
void surfelwarp::WarpSolver::allocateJtJApplyBuffer()
{
	m_apply_jtj_handler = std::make_shared<ApplyJtJHandlerMatrixFree>();
	m_apply_jtj_handler->AllocateBuffer();
}

void surfelwarp::WarpSolver::releaseJtJApplyBuffer()
{
	m_apply_jtj_handler->ReleaseBuffer();
}


/* The method to compute residual
 */
void surfelwarp::WarpSolver::allocateResidualEvaluatorBuffer() {
	m_residual_evaluator = std::make_shared<ResidualEvaluator>();
	m_residual_evaluator->AllocateBuffer();
}

void surfelwarp::WarpSolver::releaseResidualEvaluatorBuffer() {
	m_residual_evaluator->ReleaseBuffer();
}

float surfelwarp::WarpSolver::ComputeTotalResidualSynced(cudaStream_t stream) {
	return m_residual_evaluator->ComputeTotalResidualSynced(stream);
}




