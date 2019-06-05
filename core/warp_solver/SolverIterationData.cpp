//
// Created by wei on 4/22/18.
//

#include "common/Constants.h"
#include "common/ConfigParser.h"
#include "core/warp_solver/SolverIterationData.h"

/* The method for construction/destruction, buffer management
 */
surfelwarp::SolverIterationData::SolverIterationData() {
	allocateBuffer();
	m_updated_se3 = IterationInputFrom::WarpFieldInit;
	m_newton_iters = 0;

	//The flag for density and foreground term
	const auto& config = ConfigParser::Instance();
	m_use_density = config.use_density_term();
	m_use_foreground = config.use_foreground_term();
}

surfelwarp::SolverIterationData::~SolverIterationData() {
	releaseBuffer();
}

void surfelwarp::SolverIterationData::allocateBuffer() {
	node_se3_0_.AllocateBuffer(Constants::kMaxNumNodes);
	node_se3_1_.AllocateBuffer(Constants::kMaxNumNodes);
	m_twist_update.AllocateBuffer(6 * Constants::kMaxNumNodes);
}

void surfelwarp::SolverIterationData::releaseBuffer() {
	node_se3_0_.ReleaseBuffer();
	node_se3_1_.ReleaseBuffer();
	m_twist_update.ReleaseBuffer();
}


/* The processing interface
 */
void surfelwarp::SolverIterationData::SetWarpFieldInitialValue(DeviceArrayView<DualQuaternion> init_node_se3) {
	node_se3_init_ = init_node_se3;
	m_updated_se3 = IterationInputFrom::WarpFieldInit;
	m_newton_iters = 0;
	
	//Correct the size of everything
	const auto num_nodes = init_node_se3.Size();
	node_se3_0_.ResizeArrayOrException(num_nodes);
	node_se3_1_.ResizeArrayOrException(num_nodes);
	m_twist_update.ResizeArrayOrException(6 * num_nodes);

	//Init the penalty constants
	setElasticPenaltyValue(0, m_penalty_constants);
}

surfelwarp::DeviceArrayView<surfelwarp::DualQuaternion> surfelwarp::SolverIterationData::CurrentWarpFieldInput() const {
	switch(m_updated_se3) {
	case IterationInputFrom::WarpFieldInit:
		return node_se3_init_;
	case IterationInputFrom::SE3_Buffer_0:
		return node_se3_0_.ArrayView();
	case IterationInputFrom::SE3_Buffer_1:
		return node_se3_1_.ArrayView();
	default:
		LOG(FATAL) << "Should never happen";
	}
}

surfelwarp::DeviceArraySlice<float> surfelwarp::SolverIterationData::CurrentWarpFieldUpdateBuffer() {
	return m_twist_update.ArraySlice();
}

void surfelwarp::SolverIterationData::SanityCheck() const {
	const auto num_nodes = node_se3_init_.Size();
	SURFELWARP_CHECK_EQ(num_nodes, node_se3_0_.ArraySize());
	SURFELWARP_CHECK_EQ(num_nodes, node_se3_1_.ArraySize());
	SURFELWARP_CHECK_EQ(num_nodes * 6, m_twist_update.ArraySize());
}

void surfelwarp::SolverIterationData::updateIterationFlags() {
	//Update the flag
	if(m_updated_se3 == IterationInputFrom::SE3_Buffer_0) {
		m_updated_se3 = IterationInputFrom::SE3_Buffer_1;
	} else {
		//Either init or from buffer 1
		m_updated_se3 = IterationInputFrom::SE3_Buffer_0;
	}
	
	//Update the iteration counter
	m_newton_iters++;
	
	//The penalty for next iteration
	setElasticPenaltyValue(m_newton_iters, m_penalty_constants, m_use_density, m_use_foreground);
}


void surfelwarp::SolverIterationData::setElasticPenaltyValue(
	int newton_iter, 
	PenaltyConstants& constants,
	bool use_density,
	bool use_foreground
) {
	if(!Constants::kUseElasticPenalty) {
		constants.setDefaultValue();
		return;
	}

	if(newton_iter < Constants::kNumGlobalSolverItarations) {
		constants.setGlobalIterationValue(use_foreground);
	} else {
		constants.setLocalIterationValue(use_density);
	}
}

