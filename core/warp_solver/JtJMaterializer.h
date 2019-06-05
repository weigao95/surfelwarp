//
// Created by wei on 4/15/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/DeviceBufferArray.h"
#include "common/algorithm_types.h"
#include "pcg_solver/ApplySpMVBinBlockCSR.h"
#include "core/warp_solver/solver_types.h"
#include "core/warp_solver/term_offset_types.h"
#include "core/warp_solver/Node2TermsIndex.h"
#include "core/warp_solver/NodePair2TermsIndex.h"
#include "core/warp_solver/PenaltyConstants.h"
#include <memory>

namespace surfelwarp {

	class JtJMaterializer {
	private:
		//The map from term to jacobian(and residual)
		Term2JacobianMaps m_term2jacobian_map;
		
		//The map from node to terms
		using Node2TermMap = Node2TermsIndex::Node2TermMap;
		Node2TermMap m_node2term_map;
		
		//The map from node pair to terms
		using NodePair2TermMap = NodePair2TermsIndex::NodePair2TermMap;
		NodePair2TermMap m_nodepair2term_map;

		//The constants value
		PenaltyConstants m_penalty_constants;

	public:
		using Ptr = std::shared_ptr<JtJMaterializer>;
		JtJMaterializer();
		~JtJMaterializer() = default;
		SURFELWARP_NO_COPY_ASSIGN(JtJMaterializer);
		
		//Explicit allocation, release and input
		void AllocateBuffer();
		void ReleaseBuffer();
		
		//The global input
		void SetInputs(
			NodePair2TermMap nodepair2term,
			DenseDepthTerm2Jacobian dense_depth_term,
			NodeGraphSmoothTerm2Jacobian smooth_term,
			DensityMapTerm2Jacobian density_map_term = DensityMapTerm2Jacobian(),
			ForegroundMaskTerm2Jacobian foreground_mask_term = ForegroundMaskTerm2Jacobian(),
			Point2PointICPTerm2Jacobian sparse_feature_term = Point2PointICPTerm2Jacobian(),
			Node2TermMap node2term = Node2TermsIndex::Node2TermMap(),
			PenaltyConstants constants = PenaltyConstants()
		);
		
		//The processing method
		void BuildMaterializedJtJNondiagonalBlocks(cudaStream_t stream = 0);
		void BuildMaterializedJtJNondiagonalBlocksGlobalIteration(cudaStream_t stream = 0);
		
		/* The buffer for non-diagonal blocked
		 */
	private:
		DeviceBufferArray<float> m_nondiag_blks;
		void updateScalarCostJtJBlockHost(std::vector<float>& jtj_flatten, ScalarCostTerm2Jacobian term2jacobian, const float term_weight_square = 1.0f);
		void updateSmoothCostJtJBlockHost(std::vector<float>& jtj_flatten);
		void updateFeatureCostJtJBlockHost(std::vector<float>& jtj_flatten);
		void nonDiagonalBlocksSanityCheck();
	public:
		void computeNonDiagonalBlocks(cudaStream_t stream = 0);
		void computeNonDiagonalBlocksNoSync(cudaStream_t stream = 0);
		
		
		/* The method to assemble Bin-Blocked csr matrix
		 */
	private:
		DeviceBufferArray<float> m_binblock_csr_data;
		ApplySpMVBinBlockCSR<6>::Ptr m_spmv_handler;
	public:
		void AssembleBinBlockCSR(DeviceArrayView<float> diagonal_blks, cudaStream_t stream = 0);
		ApplySpMVBinBlockCSR<6>::Ptr GetSpMVHandler() { return m_spmv_handler; }
		
		/* The debug method for sparse matrix vector product
		 */
	public:
		void TestSparseMV(DeviceArrayView<float> x, DeviceArrayView<float> jtj_x_result);
	};
	
}