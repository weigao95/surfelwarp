//
// Created by wei on 4/11/18.
//

#include "common/sanity_check.h"
#include "core/warp_solver/ApplyJtJMatrixFreeHandler.h"

void surfelwarp::ApplyJtJHandlerMatrixFree::AllocateBuffer() {
	m_jacobian_dot_x.AllocateBuffer(kMaxNumScalarResidualTerms);
}

void surfelwarp::ApplyJtJHandlerMatrixFree::ReleaseBuffer() {
	m_jacobian_dot_x.ReleaseBuffer();
}


void surfelwarp::ApplyJtJHandlerMatrixFree::SetInputs(
	Node2TermMap node2term, 
	DenseDepthTerm2Jacobian dense_depth_term, 
	NodeGraphSmoothTerm2Jacobian smooth_term, 
	DensityMapTerm2Jacobian density_map_term, 
	ForegroundMaskTerm2Jacobian foreground_mask_term, 
	Point2PointICPTerm2Jacobian sparse_feature_term,
	PenaltyConstants constants
) {
	m_node2term_map = node2term;
	
	m_term2jacobian_map.dense_depth_term = dense_depth_term;
	m_term2jacobian_map.smooth_term = smooth_term;
	m_term2jacobian_map.density_map_term = density_map_term;
	m_term2jacobian_map.foreground_mask_term = foreground_mask_term;
	m_term2jacobian_map.sparse_feature_term = sparse_feature_term;
	
	m_penalty_constants = constants;
}

void surfelwarp::ApplyJtJHandlerMatrixFree::ApplyJtJ(DeviceArrayView<float> x, DeviceArraySlice<float> jtj_dot_x, cudaStream_t stream)
{
	ApplyJtJIndexed(x, jtj_dot_x, stream);
	//ApplyJtJSeparate(x, jtj_dot_x, stream);
	//ApplyJtJAtomic(x, jtj_dot_x, stream);
	
	//Sync and check error
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}

void surfelwarp::ApplyJtJHandlerMatrixFree::ApplySpMV(DeviceArrayView<float> x, DeviceArraySlice<float> spmv_x, cudaStream_t stream) {
	ApplyJtJ(x, spmv_x, stream);
}

/* These are method for sanity check
 */
void surfelwarp::ApplyJtJHandlerMatrixFree::TestApplyJtJ()
{
	//Construct random vectors
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	std::vector<float> x_h;
	x_h.resize(num_nodes * 6);
	fillRandomVector(x_h);
	
	//Upload to device
	DeviceArray<float> x_dev, jtj_x_dev;
	x_dev.create(num_nodes * 6);
	jtj_x_dev.create(num_nodes * 6);
	x_dev.upload(x_h);
	
	//Prepare for input
	DeviceArrayView<float> x_dev_view(x_dev.ptr(), x_dev.size());
	DeviceArraySlice<float> jtj_x_slice(jtj_x_dev.ptr(), jtj_x_dev.size());
	ApplyJtJIndexed(x_dev_view, jtj_x_slice);
	//ApplyJtJSeparate(x_dev_view, jtj_x_slice);
	//ApplyJtJAtomic(x_dev_view, jtj_x_slice);
	
	//Test it
	applyJtJSanityCheck(x_dev_view, jtj_x_slice.ArrayView());
	
	//Test the host implementation
	//testHostJtJ();
}













