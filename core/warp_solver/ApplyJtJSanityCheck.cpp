#include "common/logging.h"
#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/geometry_icp_jacobian.cuh"
#include "core/warp_solver/ApplyJtJMatrixFreeHandler.h"
#include "common/sanity_check.h"


void surfelwarp::ApplyJtJHandlerMatrixFree::applyJtJSanityCheck(DeviceArrayView<float> x, DeviceArrayView<float> jtj_dot_x)
{
	//LOG(INFO) << "Test GPU Implementation of Apply JtJ to a Vector";
	
	//Check the size
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	const auto desired_size = 6 * num_nodes;
	SURFELWARP_CHECK_EQ(x.Size(), desired_size);
	SURFELWARP_CHECK_EQ(jtj_dot_x.Size(), desired_size);
	
	//Download it
	std::vector<float> x_h, jtj_dot_x_dev;
	x.Download(x_h);
	jtj_dot_x.Download(jtj_dot_x_dev);
	
	//Apply JtJ directly
	std::vector<float> jtj_x_direct;
	jtj_x_direct.resize(x_h.size());
	memset(&jtj_x_direct[0], 0, sizeof(float) * jtj_x_direct.size());
	updateScalarJtJDotXDirect(x_h, m_term2jacobian_map.dense_depth_term, jtj_x_direct);
	updateSmoothJtJDotXDirect(x_h, jtj_x_direct);
	updateScalarJtJDotXDirect(x_h, m_term2jacobian_map.density_map_term, jtj_x_direct, lambda_density_square);
	updateScalarJtJDotXDirect(x_h, m_term2jacobian_map.foreground_mask_term, jtj_x_direct, lambda_foreground_square);
	updateFeatureJtJDotXDirect(x_h, jtj_x_direct);

	//Compute the relative error
	float max_relative_err = 0.0f;
	unsigned max_err_idx = 0;
	const auto& vec_0 = jtj_x_direct;
	const auto& vec_1 = jtj_dot_x_dev;
	for (auto j = 0; j < std::min(vec_0.size(), vec_1.size()); j++) {
		float value_0 = vec_0[j];
		float value_1 = vec_1[j];
		float err = std::abs(value_0 - value_1);
		if (err > 1e-3) {
			float relative_err = std::abs(err / vec_0[j]);
			if (relative_err > max_relative_err) {
				max_relative_err = relative_err;
				max_err_idx = j;
			}
		}
	}
	LOG(INFO) << "The relative error of applying JtJ is " << max_relative_err << " for element value " << vec_0[max_err_idx] << " and " << vec_1[max_err_idx];
}


void surfelwarp::ApplyJtJHandlerMatrixFree::testHostJtJ() {
	LOG(INFO) << "Test of Apply JtJ to a vector using Eigen and Direct";
	
	//Check the size
	const auto num_nodes = m_node2term_map.offset.Size() - 1;
	const auto desired_size = 6 * num_nodes;
	
	//Prepare input
	std::vector<float> x_h;
	x_h.resize(desired_size);
	fillRandomVector(x_h);
	
	//Compute the size
	auto num_depth_terms = m_node2term_map.term_offset.DenseImageTermSize();
	auto num_smooth_scalar_terms = 3 * m_node2term_map.term_offset.SmoothTermSize();
	auto num_density_map_terms = 0;
	auto num_foreground_terms = m_node2term_map.term_offset.ForegroundTermSize();
	auto num_feature_scalar_terms = 3 * m_node2term_map.term_offset.FeatureTermSize();
	auto total_terms = num_depth_terms + num_smooth_scalar_terms + num_density_map_terms + num_foreground_terms + num_feature_scalar_terms;
	
	//Using eigen to perform JtJ x
	std::vector<Eigen::Triplet<float>> jacobian_triplet;
	jacobian_triplet.clear();
	unsigned offset = 0;
	appendScalarCostJacobianTriplet(m_term2jacobian_map.dense_depth_term, offset, jacobian_triplet);
	offset += num_depth_terms;
	appendSmoothJacobianTriplet(jacobian_triplet);
	offset += num_smooth_scalar_terms;
	appendScalarCostJacobianTriplet(m_term2jacobian_map.density_map_term, offset, jacobian_triplet, lambda_density);
	offset += num_density_map_terms;
	appendScalarCostJacobianTriplet(m_term2jacobian_map.foreground_mask_term, offset, jacobian_triplet, lambda_foreground);
	offset += num_foreground_terms;
	//offset = 0;
	//total_terms = num_feature_scalar_terms;
	appendFeatureJacobianTriplet(offset, jacobian_triplet);
	
	//Apply JtJ using Eigen
	std::vector<float> jtj_x_eigen;
	jtj_x_eigen.resize(x_h.size());
	applyJtJEigen(x_h, total_terms, jacobian_triplet, jtj_x_eigen);
	
	//Apply JtJ directly
	std::vector<float> jtj_x_direct;
	jtj_x_direct.resize(jtj_x_eigen.size());
	memset(&jtj_x_direct[0], 0, sizeof(float) * jtj_x_direct.size());
	updateScalarJtJDotXDirect(x_h, m_term2jacobian_map.dense_depth_term, jtj_x_direct);
	updateSmoothJtJDotXDirect(x_h, jtj_x_direct);
	updateScalarJtJDotXDirect(x_h, m_term2jacobian_map.density_map_term, jtj_x_direct, lambda_density_square);
	updateScalarJtJDotXDirect(x_h, m_term2jacobian_map.foreground_mask_term, jtj_x_direct, lambda_foreground_square);
	updateFeatureJtJDotXDirect(x_h, jtj_x_direct);
	
	//Compute the relative error
	auto relative_err = maxRelativeError(jtj_x_direct, jtj_x_eigen);
	LOG(INFO) << "The relative error of applying JtJ to random x " << relative_err;
}