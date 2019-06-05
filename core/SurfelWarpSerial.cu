#include "common/ConfigParser.h"
#include "common/data_transfer.h"
#include "common/common_texture_utils.h"
#include "core/SurfelWarpSerial.h"
#include "core/warp_solver/WarpSolver.h"
#include "core/geometry/WarpFieldUpdater.h"
#include "core/geometry/SurfelNodeDeformer.h"
#include "visualization/Visualizer.h"
#include <thread>

/*
void surfelwarp::SurfelWarpSerial::ProcessNextFrameLegacySolver() {
	LOG(INFO) << "Current frame is " << m_frame_idx << " the updated geometry is " << m_updated_geometry_index;
	
	//Draw the required maps, assume the buffer is not mapped to cuda at input
	const auto num_vertex = m_surfel_geometry[m_updated_geometry_index]->NumValidSurfels();
	const float current_time = m_frame_idx - 1;
	const Matrix4f world2camera = Matrix4f::Identity();
	m_renderer->DrawSolverMapsWithRecentObservation(num_vertex, m_updated_geometry_index, current_time, world2camera);
	
	//Map to solver maps
	using SolverMaps = Renderer::SolverMaps;
	SolverMaps maps;
	m_renderer->MapSolverMapsToCuda(maps);
	m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);
	
	//Process the next depth frame
	CameraObservation observation;
	m_image_processor->ProcessFrameSerial(observation, m_frame_idx);
	
	//Move to data to solver
	const auto solver_geometry = m_surfel_geometry[m_updated_geometry_index]->LegacySolverAccess();
	const auto solver_warpfield = m_warp_field->LegacySolverInput();
	m_legacy_solver->SetInputModelWarpField(solver_geometry, solver_warpfield);
	
	//Prepare maps
	DeviceArray2D<float4> depth_vertex_map, depth_normal_map;
	unsigned width, height;
	query2DTextureExtent(observation.vertex_config_map, width, height);
	depth_vertex_map.create(height, width);
	depth_normal_map.create(height, width);
	textureToMap2D<float4>(observation.vertex_config_map, depth_vertex_map);
	textureToMap2D<float4>(observation.normal_radius_map, depth_normal_map);
	m_legacy_solver->SetInputMap(
		maps.reference_vertex_map,
		maps.reference_normal_map,
		maps.index_map,
		observation,
		depth_vertex_map,
		depth_normal_map,
		Eigen::Isometry3f::Identity()
	);
	m_legacy_solver->Solve(5);
	
	
	m_warp_field->SyncNodeSE3ToHost();
	SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(*m_warp_field, *m_surfel_geometry[m_updated_geometry_index]);
	
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
	
	//Hand in the input to fuser
	const auto warpfield_input = m_warp_field->GeometryUpdaterAccess();
	m_live_geometry_updater->SetInputs(
		fusion_maps,
		observation,
		warpfield_input,
		m_live_nodes_knn_skinner,
		m_updated_geometry_index,
		m_frame_idx,
		mat34(world2camera)
	);
	
	//Do fusion
	unsigned num_remaining_surfel, num_appended_surfel;
	m_live_geometry_updater->ProcessFusionSerial(num_remaining_surfel, num_appended_surfel);
	
	//Do a inverse warping
	const auto fused_geometry_idx = (m_updated_geometry_index + 1) % 2;
	SurfelNodeDeformer::InverseWarpSurfels(*m_warp_field, *m_surfel_geometry[fused_geometry_idx]);
	
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
	
	//Do a forward warp?
	//SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(*m_warp_field, *m_surfel_geometry[fused_geometry_idx]);
	
	//Debug draw
	auto draw_func = [&]()->void {
		auto geometry = m_surfel_geometry[m_updated_geometry_index]->Geometry();
		//The drawing method for solver
		auto vertex_map = observation.vertex_config_map;
		auto live_cloud = DeviceArray<float4>(geometry.live_vertex_confid.RawPtr(), geometry.live_vertex_confid.Size());
		Visualizer::DrawMatchedCloudPair(vertex_map, live_cloud, Eigen::Matrix4f::Identity());
		
		//The drawing of current geometry
		Visualizer::DrawPointCloud(live_cloud);
		
		//Draw of colored cloud
		Visualizer::DrawColoredPointCloud(geometry.live_vertex_confid.ArrayView(), geometry.color_time.ArrayView());
		
		//Also draw the matched colored cloud pair
		Visualizer::DrawMatchedCloudPair(
			observation.vertex_config_map, observation.color_time_map,
			geometry.live_vertex_confid.ArrayView(), geometry.color_time.ArrayView(),
			Eigen::Matrix4f::Identity()
		);
	};
	
	if(m_frame_idx == 159) {
		std::thread draw_thread(draw_func);
		draw_thread.join();
	}
	
	//Unmap attributes
	m_renderer->UnmapFusionMapsFromCuda();
	m_renderer->UnmapSurfelGeometryFromCuda(0);
	m_renderer->UnmapSurfelGeometryFromCuda(1);
	
	//Update the index
	m_frame_idx++;
	m_updated_geometry_index = fused_geometry_idx;
}
*/