#include "common/ConfigParser.h"
#include "common/data_transfer.h"
#include "math/eigen_device_tranfer.h"
#include "core/SurfelWarpSerial.h"
#include "core/warp_solver/WarpSolver.h"
#include "core/geometry/SurfelNodeDeformer.h"
#include "visualization/Visualizer.h"
#include <thread>
#include <chrono>

void surfelwarp::SurfelWarpSerial::TestSolverWithRigidTransform() {
	Eigen::AngleAxisf rot_x(1.2, Eigen::Vector3f::UnitX());
	Eigen::AngleAxisf rot_y(1.3, Eigen::Vector3f::UnitY());
	Eigen::Isometry3f isometry3f(rot_x * rot_y);
	//isometry3f = Eigen::Isometry3f::Identity();
	Eigen::Matrix4f camera2world = isometry3f.matrix();
	Eigen::Matrix4f world2camera = isometry3f.inverse().matrix();
	
	//Process the next frame
	CameraObservation observation;
	m_image_processor->ProcessFrameSerial(observation, m_frame_idx);
	
	m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);
	m_surfel_geometry[m_updated_geometry_index]->AddSE3ToVertexAndNormalDebug(mat34(camera2world));
	m_renderer->UnmapSurfelGeometryFromCuda(m_updated_geometry_index);
	
	//Drawing
	const auto num_vertex = m_surfel_geometry[m_updated_geometry_index]->NumValidSurfels();
	const float current_time = m_frame_idx;
	m_renderer->DrawSolverMapsWithRecentObservation(num_vertex, m_updated_geometry_index, current_time, world2camera);
	
	//Map to solver maps
	using SolverMaps = Renderer::SolverMaps;
	SolverMaps maps;
	m_renderer->MapSolverMapsToCuda(maps);
	m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);
	
	
	//The resource from geometry attributes
	const auto solver_geometry = m_surfel_geometry[m_updated_geometry_index]->SolverAccess();
	const auto solver_warpfield = m_warp_field->SolverAccess();
	
	
	//Pass the input to warp solver
	m_warp_solver->SetSolverInputs(observation, maps, solver_geometry, solver_warpfield, world2camera);
	
	//Solve it
	m_warp_solver->SolveSerial();
	const auto solved_se3 = m_warp_solver->SolvedNodeSE3();
	
	//Do a forward warp
	m_warp_field->UpdateHostDeviceNodeSE3NoSync(solved_se3);
	SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(*m_warp_field, *m_surfel_geometry[m_updated_geometry_index]);
	
	//Draw it, with new threads as opengl context are thread-wise
	auto draw_func = [&]()->void {
		//Apply the warp field for debug
		auto geometry = m_surfel_geometry[m_updated_geometry_index]->Geometry();
		
		auto vertex_map = observation.vertex_config_map;
		auto live_cloud = DeviceArray<float4>(geometry.live_vertex_confid.RawPtr(), geometry.live_vertex_confid.Size());
		Visualizer::DrawMatchedCloudPair(vertex_map, live_cloud, camera2world);
	};

	std::thread draw_thread(draw_func);
	draw_thread.join();
	
	//Draw the map for point fusion
	m_renderer->UnmapSurfelGeometryFromCuda(m_updated_geometry_index);
	m_renderer->UnmapSolverMapsFromCuda();
}


void surfelwarp::SurfelWarpSerial::TestGeometryProcessing() {
	//Draw the required maps, assume the buffer is not mapped to cuda at input
	const auto num_vertex = m_surfel_geometry[m_updated_geometry_index]->NumValidSurfels();
	const float current_time = m_frame_idx - 1;
	const Matrix4f world2camera = Matrix4f::Identity();
	m_renderer->DrawSolverMapsWithRecentObservation(num_vertex, m_updated_geometry_index, current_time, world2camera);
	
	//Map to solver maps
	using SolverMaps = Renderer::SolverMaps;
	SolverMaps solver_maps;
	m_renderer->MapSolverMapsToCuda(solver_maps);
	m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);
	
	//Process the next depth frame
	CameraObservation observation;
	m_image_processor->ProcessFrameSerial(observation, m_frame_idx + 60);
	
	//The resource from geometry attributes
	const auto solver_geometry = m_surfel_geometry[m_updated_geometry_index]->SolverAccess();
	const auto solver_warpfield = m_warp_field->SolverAccess();
	
	//Pass the input to warp solver
	m_warp_solver->SetSolverInputs(observation, solver_maps, solver_geometry, solver_warpfield, world2camera);
	
	//Solve it
	m_warp_solver->SolveSerial();
	const auto solved_se3 = m_warp_solver->SolvedNodeSE3();
	
	//Do a forward warp and build index
	m_warp_field->UpdateHostDeviceNodeSE3NoSync(solved_se3);
	SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(*m_warp_field, *m_surfel_geometry[m_updated_geometry_index], solved_se3);
	
	const auto live_nodes = m_warp_field->LiveNodeCoordinates();
	m_live_nodes_knn_skinner->BuildIndex(live_nodes);
	
	//Draw the map for point fusion
	m_renderer->UnmapSurfelGeometryFromCuda(m_updated_geometry_index);
	m_renderer->UnmapSolverMapsFromCuda();
	m_renderer->DrawFusionMaps(num_vertex, m_updated_geometry_index, world2camera);
	
	//Map the fusion map to cuda
	using FusionMaps = Renderer::FusionMaps;
	FusionMaps fusion_maps;
	m_renderer->MapFusionMapsToCuda(fusion_maps);
	//Map both maps to surfelwarp as they are both required
	m_renderer->MapSurfelGeometryToCuda(0);
	m_renderer->MapSurfelGeometryToCuda(1);
	
	//Hand in the input to reinit processor
	m_geometry_reinit_processor->SetInputs(
		fusion_maps,
		observation,
		m_updated_geometry_index,
		current_time,
		mat34(world2camera)
	);
	
	//Process it
	unsigned num_remaining_surfel, num_appended_surfel;
	m_geometry_reinit_processor->ProcessReinitObservedOnlySerial(num_remaining_surfel, num_appended_surfel);
	
	//Process on the fused geometry
	const auto fused_geometry_idx = (m_updated_geometry_index + 1) % 2;
	
	//Reinit the warp field
	const auto reference_vertex = m_surfel_geometry[fused_geometry_idx]->GetReferenceVertexConfidence();
	m_warpfield_initializer->InitializeReferenceNodeAndSE3FromVertex(reference_vertex, m_warp_field);
	
	//Build the index and skinning nodes and surfels
	m_warp_field->BuildNodeGraph();
	
	//Build skinning index
	const auto& reference_nodes = m_warp_field->ReferenceNodeCoordinates();
	m_reference_knn_skinner->BuildInitialSkinningIndex(reference_nodes);
	
	//Perform skinning
	auto skinner_geometry = m_surfel_geometry[fused_geometry_idx]->SkinnerAccess();
	auto skinner_warpfield = m_warp_field->SkinnerAccess();
	m_reference_knn_skinner->PerformSkinning(skinner_geometry, skinner_warpfield);
	
	//Debug draw
	auto draw_func = [&]()->void {
		auto geometry = m_surfel_geometry[m_updated_geometry_index]->Geometry();
		
		//The drawing method for solver
		auto vertex_map = observation.vertex_config_map;
		auto live_cloud = DeviceArray<float4>(geometry.live_vertex_confid.RawPtr(), geometry.live_vertex_confid.Size());
		Visualizer::DrawMatchedCloudPair(vertex_map, live_cloud, Eigen::Matrix4f::Identity());
		
		//Draw the fused indicator
		const auto remaining_indicator = m_geometry_reinit_processor->GetRemainingSurfelIndicator();
		const auto append_indicator = m_geometry_reinit_processor->GetAppendedSurfelIndicator();
		Visualizer::DrawFusedSurfelCloud(geometry.live_vertex_confid.ArrayView(), remaining_indicator);
		/*Visualizer::DrawFusedAppendedSurfelCloud(
			geometry.live_vertex_confid.ArrayView(),
			remaining_indicator,
			observation.vertex_config_map,
			append_indicator,
			world2camera
		);*/
		
		
		auto fused_geometry = m_surfel_geometry[fused_geometry_idx]->Geometry();
		//Visualizer::DrawPointCloud(fused_geometry.live_vertex_confid.ArrayView());
		Visualizer::DrawPointCloudWithNormal(fused_geometry.live_vertex_confid.ArrayView(), fused_geometry.live_normal_radius.ArrayView());
		Visualizer::DrawColoredPointCloud(fused_geometry.reference_vertex_confid.ArrayView(), fused_geometry.color_time.ArrayView());
		Visualizer::DrawFusedSurfelCloud(fused_geometry.live_vertex_confid.ArrayView(), num_remaining_surfel);
		
		//Also draw the matched colored cloud pair
		/*Visualizer::DrawMatchedCloudPair(
			observation.vertex_config_map, observation.color_time_map,
			geometry.live_vertex_confid.ArrayView(), geometry.color_time.ArrayView(),
			Eigen::Matrix4f::Identity()
		);*/
	};
	
	std::thread draw_thread(draw_func);
	draw_thread.join();
	
	//Unmap attributes
	m_renderer->UnmapFusionMapsFromCuda();
	m_renderer->UnmapSurfelGeometryFromCuda(0);
	m_renderer->UnmapSurfelGeometryFromCuda(1);
}


void surfelwarp::SurfelWarpSerial::ProcessNextFrameNoReinit() {
	LOG(INFO) << "Current frame is " << m_frame_idx << " the updated geometry is " << m_updated_geometry_index;
	
	//Draw the required maps, assume the buffer is not mapped to cuda at input
	const auto num_vertex = m_surfel_geometry[m_updated_geometry_index]->NumValidSurfels();
	const float current_time = m_frame_idx - 1;
	const Matrix4f init_world2camera = m_camera.GetWorld2CameraEigen();
	
	//Check the frame and draw
	SURFELWARP_CHECK(m_frame_idx >= m_reinit_frame_idx);
	const bool draw_recent = shouldDrawRecentObservation();
	if(draw_recent) {
		m_renderer->DrawSolverMapsWithRecentObservation(num_vertex, m_updated_geometry_index, current_time, init_world2camera);
	} 
	else {
		m_renderer->DrawSolverMapsConfidentObservation(num_vertex, m_updated_geometry_index, current_time, init_world2camera);
	}
	
	//Map to solver maps
	Renderer::SolverMaps solver_maps;
	m_renderer->MapSolverMapsToCuda(solver_maps);
	m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);
	
	//Process the next depth frame
	CameraObservation observation;
	m_image_processor->ProcessFrameSerial(observation, m_frame_idx);
	
	//First perform rigid solver
	m_rigid_solver->SetInputMaps(solver_maps, observation, m_camera.GetWorld2Camera());
	const mat34 solved_world2camera = m_rigid_solver->Solve();
	m_camera.SetWorld2Camera(solved_world2camera);
	
	//The resource from geometry attributes
	const auto solver_geometry = m_surfel_geometry[m_updated_geometry_index]->SolverAccess();
	const auto solver_warpfield = m_warp_field->SolverAccess();
	
	//Pass the input to warp solver
	m_warp_solver->SetSolverInputs(
		observation,
		solver_maps,
		solver_geometry,
		solver_warpfield,
		m_camera.GetWorld2Camera() //The world to camera might be updated by rigid solver
	);
	
	//Solve it
	m_warp_solver->SolveSerial();
	const auto solved_se3 = m_warp_solver->SolvedNodeSE3();
	
	//Do a forward warp and build index
	m_warp_field->UpdateHostDeviceNodeSE3NoSync(solved_se3);
	SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(*m_warp_field, *m_surfel_geometry[m_updated_geometry_index], solved_se3);
	
	//Build the live node index for later used
	const auto live_nodes = m_warp_field->LiveNodeCoordinates();
	m_live_nodes_knn_skinner->BuildIndex(live_nodes);
	
	//Draw the map for point fusion
	m_renderer->UnmapSurfelGeometryFromCuda(m_updated_geometry_index);
	m_renderer->UnmapSolverMapsFromCuda();
	m_renderer->DrawFusionMaps(num_vertex, m_updated_geometry_index, m_camera.GetWorld2CameraEigen());
	
	//Map the fusion map to cuda
	Renderer::FusionMaps fusion_maps;
	m_renderer->MapFusionMapsToCuda(fusion_maps);
	//Map both maps to surfelwarp as they are both required
	m_renderer->MapSurfelGeometryToCuda(0);
	m_renderer->MapSurfelGeometryToCuda(1);
	
	//The hand tune variable now. Should be replaced later
	const bool use_reinit = shouldDoReinit();
	const bool do_integrate = shouldDoIntegration();
	
	//The geometry index that both fusion and reinit will write to
	const auto fused_geometry_idx = (m_updated_geometry_index + 1) % 2;
	
	//Depends on should do reinit or integrate
	if(use_reinit) {
		//First setup the idx
		m_reinit_frame_idx = m_frame_idx;
		
		//Hand in the input to reinit processor
		m_geometry_reinit_processor->SetInputs(
			fusion_maps,
			observation,
			m_updated_geometry_index,
			float(m_frame_idx),
			m_camera.GetWorld2Camera()
		);
		
		//Process it
		unsigned num_remaining_surfel, num_appended_surfel;
		m_geometry_reinit_processor->ProcessReinitObservedOnlySerial(num_remaining_surfel, num_appended_surfel);
		
		//Reinit the warp field
		const auto reference_vertex = m_surfel_geometry[fused_geometry_idx]->GetReferenceVertexConfidence();
		m_warpfield_initializer->InitializeReferenceNodeAndSE3FromVertex(reference_vertex, m_warp_field);
		
		//Build the index and skinning nodes and surfels
		m_warp_field->BuildNodeGraph();
		
		//Build skinning index
		const auto& reference_nodes = m_warp_field->ReferenceNodeCoordinates();
		m_reference_knn_skinner->BuildInitialSkinningIndex(reference_nodes);
		
		//Perform skinning
		auto skinner_geometry = m_surfel_geometry[fused_geometry_idx]->SkinnerAccess();
		auto skinner_warpfield = m_warp_field->SkinnerAccess();
		m_reference_knn_skinner->PerformSkinning(skinner_geometry, skinner_warpfield);
	} else if(do_integrate) {
		//Hand in the input to fuser
		const auto warpfield_input = m_warp_field->GeometryUpdaterAccess();
		m_live_geometry_updater->SetInputs(
			fusion_maps,
			observation,
			warpfield_input,
			m_live_nodes_knn_skinner,
			m_updated_geometry_index,
			float(m_frame_idx),
			m_camera.GetWorld2Camera()
		);
		
		//Do fusion
		unsigned num_remaining_surfel, num_appended_surfel;
		m_live_geometry_updater->ProcessFusionSerial(num_remaining_surfel, num_appended_surfel);
		
		//Do a inverse warping
		SurfelNodeDeformer::InverseWarpSurfels(*m_warp_field, *m_surfel_geometry[fused_geometry_idx], solved_se3);
		
		//Extend the warp field reference nodes and SE3
		const auto prev_node_size = m_warp_field->CheckAndGetNodeSize();
		const float4* appended_vertex_ptr = m_surfel_geometry[fused_geometry_idx]->ReferenceVertexArray().RawPtr() + num_remaining_surfel;
		DeviceArrayView<float4> appended_vertex_view(appended_vertex_ptr, num_appended_surfel);
		const ushort4* appended_knn_ptr = m_surfel_geometry[fused_geometry_idx]->SurfelKNNArray().RawPtr() + num_remaining_surfel;
		DeviceArrayView<ushort4> appended_surfel_knn(appended_knn_ptr, num_appended_surfel);
		m_warpfield_extender->ExtendReferenceNodesAndSE3Sync(appended_vertex_view, appended_surfel_knn, m_warp_field);
		
		//Rebuild the node graph
		m_warp_field->BuildNodeGraph();
		
		//Update skinning
		if(m_warp_field->CheckAndGetNodeSize() > prev_node_size){
			m_reference_knn_skinner->UpdateBruteForceSkinningIndexWithNewNodes(m_warp_field->ReferenceNodeCoordinates().DeviceArrayReadOnly(), prev_node_size);
			
			//Update skinning
			auto skinner_geometry = m_surfel_geometry[fused_geometry_idx]->SkinnerAccess();
			auto skinner_warpfield = m_warp_field->SkinnerAccess();
			m_reference_knn_skinner->PerformSkinningUpdate(skinner_geometry, skinner_warpfield, prev_node_size);
		}
	}
	
	//Debug draw
	auto draw_func = [&]()->void {
		auto geometry = m_surfel_geometry[m_updated_geometry_index]->Geometry();
		
		//Draw the index map
		//Visualizer::SaveValidIndexMap(solver_maps.index_map, -1, "validity_map.png");
		
		//The drawing method for solver
		auto vertex_map = observation.vertex_config_map;
		auto live_cloud = DeviceArray<float4>(geometry.live_vertex_confid.RawPtr(), geometry.live_vertex_confid.Size());
		Visualizer::DrawMatchedCloudPair(vertex_map, live_cloud, m_camera.GetCamera2WorldEigen());
		
		//The drawing of current geometry
		Visualizer::DrawColoredPointCloud(observation.vertex_config_map, observation.color_time_map);
		
		//Draw of colored cloud
		Visualizer::DrawColoredPointCloud(geometry.live_vertex_confid.ArrayView(), geometry.color_time.ArrayView());
		
		//Also draw the matched colored cloud pair
		Visualizer::DrawMatchedCloudPair(
			observation.vertex_config_map, observation.color_time_map,
			geometry.live_vertex_confid.ArrayView(), geometry.color_time.ArrayView(),
			m_camera.GetCamera2WorldEigen()
		);
	};
	
	if(m_frame_idx == 800) {
		std::thread draw_thread(draw_func);
		draw_thread.join();
	}
	
	//Unmap attributes
	m_renderer->UnmapFusionMapsFromCuda();
	m_renderer->UnmapSurfelGeometryFromCuda(0);
	m_renderer->UnmapSurfelGeometryFromCuda(1);
	
	//Debug save
	{
		const auto with_recent = draw_recent;
		const auto& save_dir = createOrGetDataDirectory(m_frame_idx);
		saveCameraObservations(observation, save_dir);
		saveSolverMaps(solver_maps, save_dir);

		const auto num_fused_vertex = m_surfel_geometry[fused_geometry_idx]->NumValidSurfels();
		saveVisualizationMaps(
			num_fused_vertex, fused_geometry_idx,
			m_camera.GetWorld2CameraEigen(), m_camera.GetInitWorld2CameraEigen(),
			save_dir, with_recent
		);
		
		//Save the matched pair
		//if(m_frame_idx == 99) {
		//	saveCorrespondedCloud(observation, fused_geometry_idx, save_dir);
		//}
	}
	
	//Update the index
	m_frame_idx++;
	m_updated_geometry_index = fused_geometry_idx;
}

void surfelwarp::SurfelWarpSerial::TestRigidSolver() {
	Eigen::AngleAxisf rot_x(.02, Eigen::Vector3f::UnitX());
	Eigen::AngleAxisf rot_y(.05, Eigen::Vector3f::UnitY());
	Eigen::Isometry3f isometry3f(rot_x * rot_y);
	//isometry3f = Eigen::Isometry3f::Identity();
	Eigen::Matrix4f camera2world = isometry3f.matrix();
	Eigen::Matrix4f world2camera = isometry3f.inverse().matrix();
	
	//Process the next frame
	CameraObservation observation;
	m_image_processor->ProcessFrameSerial(observation, m_frame_idx);
	
	m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);
	m_surfel_geometry[m_updated_geometry_index]->AddSE3ToVertexAndNormalDebug(mat34(camera2world));
	m_renderer->UnmapSurfelGeometryFromCuda(m_updated_geometry_index);
	
	//Draw the required maps, assume the buffer is not mapped to cuda at input
	const auto num_vertex = m_surfel_geometry[m_updated_geometry_index]->NumValidSurfels();
	const float current_time = m_frame_idx - 1;
	m_renderer->DrawSolverMapsWithRecentObservation(num_vertex, m_updated_geometry_index, current_time, Matrix4f::Identity());
	
	//Map to solver maps
	using SolverMaps = Renderer::SolverMaps;
	SolverMaps solver_maps;
	m_renderer->MapSolverMapsToCuda(solver_maps);
	m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);
	
	//Use the rigid solve
	m_rigid_solver->SetInputMaps(solver_maps, observation, mat34::identity());
	const mat34 solved_world2camera = m_rigid_solver->Solve(10);
	//const mat34 solved_world2camera = mat34::identity();
	
	//Draw the matched cloud pair
	auto draw_func = [&]() -> void {
		auto geometry = m_surfel_geometry[m_updated_geometry_index]->Geometry();
		
		//The drawing method for solver
		auto vertex_map = observation.vertex_config_map;
		auto live_cloud = DeviceArray<float4>(geometry.live_vertex_confid.RawPtr(), geometry.live_vertex_confid.Size());
		Visualizer::DrawMatchedCloudPair(vertex_map, live_cloud, toEigen(solved_world2camera.inverse()));
	};

	//Draw it in another thread
	std::thread draw_thread(draw_func);
	draw_thread.join();
	
	m_renderer->UnmapSolverMapsFromCuda();
	m_renderer->UnmapSurfelGeometryFromCuda(m_updated_geometry_index);
}

void surfelwarp::SurfelWarpSerial::TestSolver() {
	Eigen::Matrix4f world2camera = m_camera.GetWorld2CameraEigen();
	
	//Process the next frame
	CameraObservation observation;
	m_image_processor->ProcessFrameSerial(observation, m_frame_idx);
	
	//Drawing
	const auto num_vertex = m_surfel_geometry[m_updated_geometry_index]->NumValidSurfels();
	const float current_time = m_frame_idx;
	m_renderer->DrawSolverMapsWithRecentObservation(num_vertex, m_updated_geometry_index, current_time, world2camera);
	
	//Map to solver maps
	using SolverMaps = Renderer::SolverMaps;
	SolverMaps maps;
	m_renderer->MapSolverMapsToCuda(maps);
	m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);
	
	
	//The resource from geometry attributes
	const auto solver_geometry = m_surfel_geometry[m_updated_geometry_index]->SolverAccess();
	const auto solver_warpfield = m_warp_field->SolverAccess();
	
	
	//Pass the input to warp solver
	m_warp_solver->SetSolverInputs(observation, maps, solver_geometry, solver_warpfield, world2camera);
	
	//Test the solver
	//m_warp_solver->MaterializedFullSolverIterationTest();
	m_warp_solver->SolveSerial();
	const auto solved_se3 = m_warp_solver->SolvedNodeSE3();
	
	//Do a forward warp
	SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(*m_warp_field, *m_surfel_geometry[m_updated_geometry_index], solved_se3);
	
	//Draw it, with new threads as opengl context are thread-wise
	auto draw_func = [&]()->void {
		//Apply the warp field for debug
		auto geometry = m_surfel_geometry[m_updated_geometry_index]->Geometry();
		
		auto vertex_map = observation.vertex_config_map;
		auto live_cloud = DeviceArray<float4>(geometry.live_vertex_confid.RawPtr(), geometry.live_vertex_confid.Size());
		Visualizer::DrawMatchedCloudPair(vertex_map, live_cloud, m_camera.GetCamera2WorldEigen());
	};
	
	std::thread draw_thread(draw_func);
	draw_thread.join();
	
	//Draw the map for point fusion
	m_renderer->UnmapSurfelGeometryFromCuda(m_updated_geometry_index);
	m_renderer->UnmapSolverMapsFromCuda();
}

void surfelwarp::SurfelWarpSerial::TestPerformance()
{
	//Draw the required maps, assume the buffer is not mapped to cuda at input
	const auto num_vertex = m_surfel_geometry[m_updated_geometry_index]->NumValidSurfels();
	const float current_time = m_frame_idx - 1;
	const Matrix4f init_world2camera = m_camera.GetWorld2CameraEigen();

	//Check the frame and draw
	SURFELWARP_CHECK(m_frame_idx >= m_reinit_frame_idx);
	const bool draw_recent = shouldDrawRecentObservation();
	if (draw_recent) {
		m_renderer->DrawSolverMapsWithRecentObservation(num_vertex, m_updated_geometry_index, current_time, init_world2camera);
	}
	else {
		m_renderer->DrawSolverMapsConfidentObservation(num_vertex, m_updated_geometry_index, current_time, init_world2camera);
	}

	//Map to solver maps
	Renderer::SolverMaps solver_maps;
	m_renderer->MapSolverMapsToCuda(solver_maps);
	m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);

	//Process the next depth frame
	CameraObservation observation;
	//m_image_processor->ProcessFrameSerial(observation, m_frame_idx);
	m_image_processor->ProcessFrameStreamed(observation, m_frame_idx);

	//First perform rigid solver
	m_rigid_solver->SetInputMaps(solver_maps, observation, m_camera.GetWorld2Camera());
	const mat34 solved_world2camera = m_rigid_solver->Solve();
	m_camera.SetWorld2Camera(solved_world2camera);

	//The resource from geometry attributes
	const auto solver_geometry = m_surfel_geometry[m_updated_geometry_index]->SolverAccess();
	const auto solver_warpfield = m_warp_field->SolverAccess();

	//Pass the input to warp solver
	m_warp_solver->SetSolverInputs(
		observation,
		solver_maps,
		solver_geometry,
		solver_warpfield,
		m_camera.GetWorld2Camera() //The world to camera might be updated by rigid solver
	);

	//Solve it
	//m_warp_solver->SolveSerial();
	m_warp_solver->SolveStreamed();
	{
		using namespace std::chrono;
		const auto time_start = high_resolution_clock::now();
		for(auto i = 0; i < 100; i++){
			m_warp_solver->SolveStreamed();
		}
		const auto time_end = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(time_end - time_start);
		std::cout << "It took " << time_span.count() << " seconds.";
	}
	const auto solved_se3 = m_warp_solver->SolvedNodeSE3();

	//Do a forward warp and build index
	m_warp_field->UpdateHostDeviceNodeSE3NoSync(solved_se3);
	SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(*m_warp_field, *m_surfel_geometry[m_updated_geometry_index], solved_se3);

	//Compute the nodewise error
	m_warp_solver->ComputeAlignmentErrorOnNodes();

	//Build the live node index for later used
	const auto live_nodes = m_warp_field->LiveNodeCoordinates();
	m_live_nodes_knn_skinner->BuildIndex(live_nodes);

	//Draw the map for point fusion
	m_renderer->UnmapSurfelGeometryFromCuda(m_updated_geometry_index);
	m_renderer->UnmapSolverMapsFromCuda();
	m_renderer->DrawFusionMaps(num_vertex, m_updated_geometry_index, m_camera.GetWorld2CameraEigen());

	//Map the fusion map to cuda
	Renderer::FusionMaps fusion_maps;
	m_renderer->MapFusionMapsToCuda(fusion_maps);
	//Map both maps to surfelwarp as they are both required
	m_renderer->MapSurfelGeometryToCuda(0);
	m_renderer->MapSurfelGeometryToCuda(1);

	//The hand tune variable now. Should be replaced later
	const bool use_reinit = shouldDoReinit();
	const bool do_integrate = shouldDoIntegration();

	//The geometry index that both fusion and reinit will write to, if no writing then keep current geometry index
	auto fused_geometry_idx = m_updated_geometry_index;

	//Depends on should do reinit or integrate
	if (use_reinit) {
		//First setup the idx
		m_reinit_frame_idx = m_frame_idx;
		fused_geometry_idx = (m_updated_geometry_index + 1) % 2;

		//Hand in the input to reinit processor
		m_geometry_reinit_processor->SetInputs(
			fusion_maps,
			observation,
			m_updated_geometry_index,
			float(m_frame_idx),
			m_camera.GetWorld2Camera()
		);

		//Process it
		const auto node_error = m_warp_solver->GetNodeAlignmentError();
		unsigned num_remaining_surfel, num_appended_surfel;
		m_geometry_reinit_processor->ProcessReinitObservedOnlySerial(num_remaining_surfel, num_appended_surfel);
		//m_geometry_reinit_processor->ProcessReinitNodeErrorSerial(num_remaining_surfel, num_appended_surfel, node_error, 0.06f);

		//Reinit the warp field
		const auto reference_vertex = m_surfel_geometry[fused_geometry_idx]->GetReferenceVertexConfidence();
		m_warpfield_initializer->InitializeReferenceNodeAndSE3FromVertex(reference_vertex, m_warp_field);

		//Build the index and skinning nodes and surfels
		m_warp_field->BuildNodeGraph();

		//Build skinning index
		const auto& reference_nodes = m_warp_field->ReferenceNodeCoordinates();
		m_reference_knn_skinner->BuildInitialSkinningIndex(reference_nodes);

		//Perform skinning
		auto skinner_geometry = m_surfel_geometry[fused_geometry_idx]->SkinnerAccess();
		auto skinner_warpfield = m_warp_field->SkinnerAccess();
		m_reference_knn_skinner->PerformSkinning(skinner_geometry, skinner_warpfield);
	}
	else if (do_integrate) {
		//Update the frame idx
		fused_geometry_idx = (m_updated_geometry_index + 1) % 2;

		//Hand in the input to fuser
		const auto warpfield_input = m_warp_field->GeometryUpdaterAccess();
		m_live_geometry_updater->SetInputs(
			fusion_maps,
			observation,
			warpfield_input,
			m_live_nodes_knn_skinner,
			m_updated_geometry_index,
			float(m_frame_idx),
			m_camera.GetWorld2Camera()
		);

		//Do fusion
		unsigned num_remaining_surfel, num_appended_surfel;
		//m_live_geometry_updater->ProcessFusionSerial(num_remaining_surfel, num_appended_surfel);
		m_live_geometry_updater->ProcessFusionStreamed(num_remaining_surfel, num_appended_surfel);

		//Do a inverse warping
		SurfelNodeDeformer::InverseWarpSurfels(*m_warp_field, *m_surfel_geometry[fused_geometry_idx], solved_se3);

		//Extend the warp field reference nodes and SE3
		const auto prev_node_size = m_warp_field->CheckAndGetNodeSize();
		const float4* appended_vertex_ptr = m_surfel_geometry[fused_geometry_idx]->ReferenceVertexArray().RawPtr() + num_remaining_surfel;
		DeviceArrayView<float4> appended_vertex_view(appended_vertex_ptr, num_appended_surfel);
		const ushort4* appended_knn_ptr = m_surfel_geometry[fused_geometry_idx]->SurfelKNNArray().RawPtr() + num_remaining_surfel;
		DeviceArrayView<ushort4> appended_surfel_knn(appended_knn_ptr, num_appended_surfel);
		m_warpfield_extender->ExtendReferenceNodesAndSE3Sync(appended_vertex_view, appended_surfel_knn, m_warp_field);

		//Rebuild the node graph
		m_warp_field->BuildNodeGraph();

		//Update skinning
		if (m_warp_field->CheckAndGetNodeSize() > prev_node_size) {
			m_reference_knn_skinner->UpdateBruteForceSkinningIndexWithNewNodes(m_warp_field->ReferenceNodeCoordinates().DeviceArrayReadOnly(), prev_node_size);

			//Update skinning
			auto skinner_geometry = m_surfel_geometry[fused_geometry_idx]->SkinnerAccess();
			auto skinner_warpfield = m_warp_field->SkinnerAccess();
			m_reference_knn_skinner->PerformSkinningUpdate(skinner_geometry, skinner_warpfield, prev_node_size);
		}
	}

	//Unmap attributes
	m_renderer->UnmapFusionMapsFromCuda();
	m_renderer->UnmapSurfelGeometryFromCuda(0);
	m_renderer->UnmapSurfelGeometryFromCuda(1);

	//Update the index
	m_frame_idx++;
	m_updated_geometry_index = fused_geometry_idx;
}
