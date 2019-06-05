//
// Created by wei on 4/26/18.
//

#include "core/warp_solver/solver_constants.h"
#include "core/warp_solver/PenaltyConstants.h"


surfelwarp::PenaltyConstants::PenaltyConstants() {
	setDefaultValue();
}

void surfelwarp::PenaltyConstants::setDefaultValue() {
#if defined(USE_DENSE_SOLVER_MAPS)
	m_lambda_smooth = 2.3f;
	m_lambda_density = 0.0f;
	m_lambda_foreground = 0.0f;
	m_lambda_feature = 0.7f;
#else
	m_lambda_smooth = 2.0f;
	m_lambda_density = 0.0f;
	m_lambda_foreground = 0.0f;
	m_lambda_feature = 0.0f;
#endif
}

void surfelwarp::PenaltyConstants::setGlobalIterationValue(bool use_foreground) {
	m_lambda_smooth = 2.3f;
	m_lambda_density = 0.0f;
	if(use_foreground)
		m_lambda_foreground = 2e-3f;
	else
		m_lambda_foreground = 0.0f;
	m_lambda_feature = 1.0f;
}

void surfelwarp::PenaltyConstants::setLocalIterationValue(bool use_density) {
	m_lambda_smooth = 2.3f;
	if(use_density)
		m_lambda_density = 1e-2f;
	else
		m_lambda_density = 0.0f;
	m_lambda_foreground = 0.0f;
	m_lambda_feature = 0.0f;
}
