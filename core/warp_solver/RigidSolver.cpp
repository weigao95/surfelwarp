//
// Created by wei on 5/22/18.
//

#include "common/ConfigParser.h"
#include "core/warp_solver/RigidSolver.h"

#include <Eigen/Eigen>

surfelwarp::RigidSolver::RigidSolver() {
	//Init the intrisic for projection
	const auto& config = ConfigParser::Instance();
	m_project_intrinsic = config.rgb_intrinsic_clip();
	m_image_rows = config.clip_image_rows();
	m_image_cols = config.clip_image_cols();
	
	//Init the world2camera
	m_curr_world2camera = mat34::identity();
	
	//Allocate the buffer
	allocateReduceBuffer();
}

surfelwarp::RigidSolver::~RigidSolver() {

}

void surfelwarp::RigidSolver::SetInputMaps(
	const surfelwarp::Renderer::SolverMaps &solver_maps,
	const surfelwarp::CameraObservation &observation,
	const mat34& init_world2camera
) {
	m_solver_maps.live_vertex_map = solver_maps.warp_vertex_map;
	m_solver_maps.live_normal_map = solver_maps.warp_normal_map;
	
	m_observation.vertex_map = observation.vertex_config_map;
	m_observation.normal_map = observation.normal_radius_map;
	
	m_curr_world2camera = init_world2camera;
}

surfelwarp::mat34 surfelwarp::RigidSolver::Solve(int max_iters, cudaStream_t stream) {
	//The solver iteration
	for(int i = 0; i < max_iters; i++) {
		rigidSolveDeviceIteration(stream);
		rigidSolveHostIterationSync(stream);
	}
	
	//The updated world2camera
	return m_curr_world2camera;
}

void surfelwarp::RigidSolver::rigidSolveHostIterationSync(cudaStream_t stream) {
	//Sync before using the data
	cudaSafeCall(cudaStreamSynchronize(stream));
	
	//Load the hsot array
	const auto& host_array = m_reduced_matrix_vector.HostArray();
	
	//Load the data into Eigen
	auto shift = 0;
	for (int i = 0; i < 6; i++) {
		for (int j = i; j < 6; j++) {
			const float value = host_array[shift++];
			JtJ_(i, j) = value;
			JtJ_(j, i) = value;
		}
	}
	for (int i = 0; i < 6; i++) {
		const float value = host_array[shift++];
		JtErr_[i] = value;
	}
	
	//Solve it
	Eigen::Matrix<float, 6, 1> x = JtJ_.llt().solve(JtErr_).cast<float>();

	//Update the se3
	const float3 twist_rot = make_float3(x(0), x(1), x(2));
	const float3 twist_trans = make_float3(x(3), x(4), x(5));
	const mat34 se3_update(twist_rot, twist_trans);
	m_curr_world2camera = se3_update * m_curr_world2camera;
}