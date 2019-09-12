//
// Created by wei on 4/15/18.
//
#include "common/ConfigParser.h"
#include "common/Constants.h"
#include "core/warp_solver/JtJMaterializer.h"

surfelwarp::JtJMaterializer::JtJMaterializer() {
	memset(&m_node2term_map, 0, sizeof(m_node2term_map));
	memset(&m_nodepair2term_map, 0, sizeof(m_nodepair2term_map));
	memset(&m_term2jacobian_map, 0, sizeof(m_term2jacobian_map));
}

void surfelwarp::JtJMaterializer::AllocateBuffer() {
	m_nondiag_blks.AllocateBuffer(36 * Constants::kMaxNumNodePairs);
	m_binblock_csr_data.AllocateBuffer(36 * (Constants::kMaxNumNodePairs + Constants::kMaxNumNodes));
	m_spmv_handler = std::make_shared<ApplySpMVBinBlockCSR<6>>();
}

void surfelwarp::JtJMaterializer::ReleaseBuffer() {
	m_nondiag_blks.ReleaseBuffer();
}

void surfelwarp::JtJMaterializer::SetInputs(
	JtJMaterializer::NodePair2TermMap nodepair2term,
	DenseDepthTerm2Jacobian dense_depth_term,
	NodeGraphSmoothTerm2Jacobian smooth_term,
	DensityMapTerm2Jacobian density_map_term,
	ForegroundMaskTerm2Jacobian foreground_mask_term,
	Point2PointICPTerm2Jacobian sparse_feature_term,
	JtJMaterializer::Node2TermMap node2term,
	PenaltyConstants constants
) {
	m_nodepair2term_map = nodepair2term;
	m_node2term_map = node2term;
	
	m_term2jacobian_map.dense_depth_term = dense_depth_term;
	m_term2jacobian_map.smooth_term = smooth_term;
	m_term2jacobian_map.density_map_term = density_map_term;
	m_term2jacobian_map.foreground_mask_term = foreground_mask_term;
	m_term2jacobian_map.sparse_feature_term = sparse_feature_term;

	m_penalty_constants = constants;
}

void surfelwarp::JtJMaterializer::BuildMaterializedJtJNondiagonalBlocks(cudaStream_t stream) {
	computeNonDiagonalBlocks(stream);
	//computeNonDiagonalBlocksNoSync(stream);

	//TODO: instead of leaving dead code, either change this function w/ optional argument/flag check or
	// use a separate set of functions / child class w/ overriding member functions for performance measurement
    //TODO: get rid of PCL usage in favor of std::chrono or the like
	//Performance test
	/*{
		pcl::ScopeTime time("Performance test of Compute JtJ");
		for(auto i = 0; i < 1000; i++) {
			computeNonDiagonalBlocks(stream);
			//computeNonDiagonalBlocksNoSync(stream);
		}
		cudaStreamSynchronize(stream);
	}*/
}





