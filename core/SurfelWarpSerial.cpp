//
// Created by wei on 3/29/18.
//

#include "common/ConfigParser.h"
#include "common/data_transfer.h"
#include "common/TimeLogger.h"
#include "core/SurfelWarpSerial.h"
#include "core/warp_solver/WarpSolver.h"
#include "core/geometry/SurfelNodeDeformer.h"
#include "visualization/Visualizer.h"
#include "imgproc/frameio/FetchInterface.h"
#include "imgproc/frameio/GenericFileFetch.h"
#include "imgproc/frameio/VolumeDeformFileFetch.h"
// #include "imgproc/frameio/AzureKinectDKFetch.h"
#include "imgproc/frameio/RealsenseFetch.h"
#include "visualization/Visualizer.h"

#include <thread>
#include <fstream>

bool surfelwarp::SurfelWarpSerial::m_use_periodic_reinit
    = surfelwarp::ConfigParser::Instance().use_periodic_reinit();

bool surfelwarp::SurfelWarpSerial::get_use_periodic_reinit()
{
    return m_use_periodic_reinit;
}

void surfelwarp::SurfelWarpSerial::set_use_periodic_reinit(bool value)
{
    m_use_periodic_reinit = value;
}

surfelwarp::SurfelWarpSerial::SurfelWarpSerial()
{
    // The config is assumed to be updated
    const auto& config = ConfigParser::Instance();

    // Construct the image processor
    FetchInterface::Ptr fetcher;

    if (config.getIOMode() == "GenericFileFetch") {
        fetcher = std::make_shared<GenericFileFetch>(config.data_path());
    } else if (config.getIOMode() == "VolumeDeformFileFetch") {
        fetcher = std::make_shared<VolumeDeformFileFetch>(config.data_path());
    } else if (config.getIOMode() == "realsense") {
        fetcher = std::make_shared<RealsenseFetch>(
            config.data_path(), config.isSaveOnlineFrame());
    } else {
        throw(
            std::runtime_error(config.getIOMode() + " io_mode not supported"));
    }
    LOG(INFO);
    m_image_processor = std::make_shared<ImageProcessor>(fetcher);

    // Construct the holder for surfel geometry
    m_surfel_geometry[0] = std::make_shared<SurfelGeometry>();
    m_surfel_geometry[1] = std::make_shared<SurfelGeometry>();
    m_live_geometry_updater
        = std::make_shared<LiveGeometryUpdater>(m_surfel_geometry);

    // The warp field
    m_warp_field = std::make_shared<WarpField>();
    m_warpfield_initializer = std::make_shared<WarpFieldInitializer>();
    m_warpfield_extender = std::make_shared<WarpFieldExtender>();

    // The knn index
    m_live_nodes_knn_skinner = KNNBruteForceLiveNodes::Instance();
    m_reference_knn_skinner = ReferenceNodeSkinner::Instance();

    // Construct the renderer
    m_renderer = std::make_shared<Renderer>(
        config.clip_image_rows(), config.clip_image_cols());

    // Map the resource into geometry
    m_renderer->MapSurfelGeometryToCuda(0, *(m_surfel_geometry[0]));
    m_renderer->MapSurfelGeometryToCuda(1, *(m_surfel_geometry[1]));
    m_renderer->UnmapSurfelGeometryFromCuda(0);
    m_renderer->UnmapSurfelGeometryFromCuda(1);
    m_updated_geometry_index = -1; // None are updated

    // Constructs the solver
    m_warp_solver = std::make_shared<WarpSolver>();
    m_rigid_solver = std::make_shared<RigidSolver>();
    m_warp_solver->AllocateBuffer();

    // Construct the initializer and re-initializer
    m_geometry_initializer = std::make_shared<GeometryInitializer>();
    m_geometry_initializer->AllocateBuffer();
    m_geometry_reinit_processor
        = std::make_shared<GeometryReinitProcessor>(m_surfel_geometry);

    // The frame index
    m_frame_idx = config.start_frame_idx();
    m_reinit_frame_idx = m_frame_idx;
}

surfelwarp::SurfelWarpSerial::~SurfelWarpSerial()
{
    // Release the explicit allocated buffer
    m_warp_solver->ReleaseBuffer();
    m_geometry_initializer->ReleaseBuffer();
}

/* The processing interface
 */
void surfelwarp::SurfelWarpSerial::ProcessFirstFrame()
{
    // Process it
    const auto surfel_array
        = m_image_processor->ProcessFirstFrameSerial(m_frame_idx);
    // Build the reference and live nodes, color and init time
    m_updated_geometry_index = 0;
    m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);
    m_geometry_initializer->InitFromObservationSerial(
        *m_surfel_geometry[m_updated_geometry_index], surfel_array);

    // Build the reference vertex and SE3 for the warp field
    const auto reference_vertex = m_surfel_geometry[m_updated_geometry_index]
                                      ->GetReferenceVertexConfidence();
    m_warpfield_initializer->InitializeReferenceNodeAndSE3FromVertex(
        reference_vertex, m_warp_field);

    // Build the index and skinning nodes and surfels
    m_warp_field->BuildNodeGraph();

    // Perform skinning
    const auto& reference_nodes = m_warp_field->ReferenceNodeCoordinates();
    m_reference_knn_skinner->BuildInitialSkinningIndex(reference_nodes);

    auto skinner_geometry
        = m_surfel_geometry[m_updated_geometry_index]->SkinnerAccess();
    auto skinner_warpfield = m_warp_field->SkinnerAccess();
    m_reference_knn_skinner->PerformSkinning(
        skinner_geometry, skinner_warpfield);

    // Unmap it
    m_renderer->UnmapSurfelGeometryFromCuda(m_updated_geometry_index);

    // Update the index
    m_frame_idx++;
}

static std::ostream& operator<<(std::ostream& os, float3 vec)
{
    os << vec.x << "\t" << vec.y << "\t" << vec.z << "\t";
    return os;
}

void surfelwarp::SurfelWarpSerial::ProcessNextFrameWithReinit(
    const ConfigParser& config)
{
    TIME_LOG(TimeLogger::printTimeLog("start to process a new frame"));
    // Draw the required maps, assume the buffer is not mapped to cuda at input
    const auto num_vertex
        = m_surfel_geometry[m_updated_geometry_index]->NumValidSurfels();
    const float current_time = m_frame_idx - 1;
    const Matrix4f init_world2camera = m_camera.GetWorld2CameraEigen();

    // Check the frame and draw
    SURFELWARP_CHECK(m_frame_idx >= m_reinit_frame_idx);
    const bool draw_recent = shouldDrawRecentObservation();
    if (draw_recent) {
        m_renderer->DrawSolverMapsWithRecentObservation(num_vertex,
            m_updated_geometry_index, current_time, init_world2camera);
    } else {
        m_renderer->DrawSolverMapsConfidentObservation(num_vertex,
            m_updated_geometry_index, current_time, init_world2camera);
    }
    TIME_LOG(TimeLogger::printTimeLog("Draw Solver Maps"));

    // Map to solver maps
    Renderer::SolverMaps solver_maps;
    m_renderer->MapSolverMapsToCuda(solver_maps);
    m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);
    TIME_LOG(TimeLogger::printTimeLog("Map to solver maps"));

    // Process the next depth frame
    CameraObservation observation;
    m_image_processor->ProcessFrameStreamed(observation, m_frame_idx);
    TIME_LOG(TimeLogger::printTimeLog("ImageProcess a frame"));

    // (xt) TODO: utilize IMU data to initialize W2C, to deal with fast camera
    // move First perform rigid solver
    m_rigid_solver->SetInputMaps(
        solver_maps, observation, m_camera.GetWorld2Camera());
    const mat34 solved_world2camera = m_rigid_solver->Solve();
    m_camera.SetWorld2Camera(solved_world2camera);
    TIME_LOG(TimeLogger::printTimeLog("rigid solve"));

    // The resource from geometry attributes
    const auto solver_geometry
        = m_surfel_geometry[m_updated_geometry_index]->SolverAccess();
    const auto solver_warpfield = m_warp_field->SolverAccess();

    // Pass the input to warp solver
    m_warp_solver->SetSolverInputs(
        observation, solver_maps, solver_geometry, solver_warpfield,
        m_camera.GetWorld2Camera() // The world to camera might be updated by
                                   // rigid solver
    );

    // Solve it
    m_warp_solver->SolveStreamed();
    const auto solved_se3 = m_warp_solver->SolvedNodeSE3();
    TIME_LOG(TimeLogger::printTimeLog("non-rigid solve"));

    if (config.save_se3) {
        // download se3
        std::vector<DualQuaternion> se3_h;
        solved_se3.Download(se3_h);

        // download coord
        std::vector<float4> coord_h;
        solver_warpfield.reference_node_coords.Download(coord_h);

        if (coord_h.size() != se3_h.size()) {
            std::cout << "Error!" << std::endl;
        }

        // save into file
        char filename[50];
        sprintf(filename, "temp/se3_frame_%06d.log", m_frame_idx);
        std::ofstream of(filename, std::ios::ios_base::out);

        for (int i = 0; i < se3_h.size(); i++) {
            // transformation(dq) of node i
            DualQuaternion dq = se3_h[i];
            // position of node i
            float4 pos = coord_h[i];
            float3 p = make_float3(pos.x, pos.y, pos.z);
            // transformation(axis_angle and trans) of node i
            float3 axis_angle, trans;
            dq.convert2(axis_angle, trans);
            // transformation(mat34) of node i
            mat34 m = se3_h[i].mat34_f();
            // displacement of node i
            float3 d = dq * p - p;
            of << se3_h[i] << coord_h[i] << axis_angle << trans << d
               << std::endl;
        }
        // for(auto it = se3_h.begin(); it != se3_h.end(); it++){
        //	of << *it << std::endl;
        // }
    }

    // Do a forward warp and build index
    m_warp_field->UpdateHostDeviceNodeSE3NoSync(solved_se3);
    SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(*m_warp_field,
        *m_surfel_geometry[m_updated_geometry_index], solved_se3);

    // Compute the nodewise error
    m_warp_solver->ComputeAlignmentErrorOnNodes();

    // Build the live node index for later used
    const auto live_nodes = m_warp_field->LiveNodeCoordinates();
    m_live_nodes_knn_skinner->BuildIndex(live_nodes);

    // Draw the map for point fusion
    m_renderer->UnmapSurfelGeometryFromCuda(m_updated_geometry_index);
    m_renderer->UnmapSolverMapsFromCuda();
    m_renderer->DrawFusionMaps(
        num_vertex, m_updated_geometry_index, m_camera.GetWorld2CameraEigen());

    // Map the fusion map to cuda
    Renderer::FusionMaps fusion_maps;
    m_renderer->MapFusionMapsToCuda(fusion_maps);
    // Map both maps to surfelwarp as they are both required
    m_renderer->MapSurfelGeometryToCuda(0);
    m_renderer->MapSurfelGeometryToCuda(1);
    TIME_LOG(TimeLogger::printTimeLog("something after non-rigid solve"));

    // The hand tune variable now. Should be replaced later
    const bool use_reinit = shouldDoReinit();
    const bool do_integrate = shouldDoIntegration();

    // The geometry index that both fusion and reinit will write to, if no
    // writing then keep current geometry index
    auto fused_geometry_idx = m_updated_geometry_index;

    // Depends on should do reinit or integrate
    if (use_reinit) {
        // First setup the idx
        m_reinit_frame_idx = m_frame_idx;
        fused_geometry_idx = (m_updated_geometry_index + 1) % 2;

        // Hand in the input to reinit processor
        m_geometry_reinit_processor->SetInputs(fusion_maps, observation,
            m_updated_geometry_index, float(m_frame_idx),
            m_camera.GetWorld2Camera());

        // Process it
        const auto node_error = m_warp_solver->GetNodeAlignmentError();
        unsigned num_remaining_surfel, num_appended_surfel;
        m_geometry_reinit_processor->ProcessReinitObservedOnlySerial(
            num_remaining_surfel, num_appended_surfel);
        // m_geometry_reinit_processor->ProcessReinitNodeErrorSerial(num_remaining_surfel,
        // num_appended_surfel, node_error, 0.06f);

        // Reinit the warp field
        const auto reference_vertex = m_surfel_geometry[fused_geometry_idx]
                                          ->GetReferenceVertexConfidence();
        m_warpfield_initializer->InitializeReferenceNodeAndSE3FromVertex(
            reference_vertex, m_warp_field);

        // Build the index and skinning nodes and surfels
        m_warp_field->BuildNodeGraph();

        // Build skinning index
        const auto& reference_nodes = m_warp_field->ReferenceNodeCoordinates();
        m_reference_knn_skinner->BuildInitialSkinningIndex(reference_nodes);

        // Perform skinning
        auto skinner_geometry
            = m_surfel_geometry[fused_geometry_idx]->SkinnerAccess();
        auto skinner_warpfield = m_warp_field->SkinnerAccess();
        m_reference_knn_skinner->PerformSkinning(
            skinner_geometry, skinner_warpfield);
        TIME_LOG(TimeLogger::printTimeLog("reinit"));
    } else if (do_integrate) {
        // Update the frame idx
        fused_geometry_idx = (m_updated_geometry_index + 1) % 2;

        // Hand in the input to fuser
        const auto warpfield_input = m_warp_field->GeometryUpdaterAccess();
        m_live_geometry_updater->SetInputs(fusion_maps, observation,
            warpfield_input, m_live_nodes_knn_skinner, m_updated_geometry_index,
            float(m_frame_idx), m_camera.GetWorld2Camera());

        // Do fusion
        unsigned num_remaining_surfel, num_appended_surfel;
        m_live_geometry_updater->ProcessFusionStreamed(
            num_remaining_surfel, num_appended_surfel);

        // Do a inverse warping
        SurfelNodeDeformer::InverseWarpSurfels(
            *m_warp_field, *m_surfel_geometry[fused_geometry_idx], solved_se3);
        // Extend the warp field reference nodes and SE3
        const auto prev_node_size = m_warp_field->CheckAndGetNodeSize();
        const float4* appended_vertex_ptr
            = m_surfel_geometry[fused_geometry_idx]
                  ->ReferenceVertexArray()
                  .RawPtr()
              + num_remaining_surfel;
        DeviceArrayView<float4> appended_vertex_view(
            appended_vertex_ptr, num_appended_surfel);
        const ushort4* appended_knn_ptr
            = m_surfel_geometry[fused_geometry_idx]->SurfelKNNArray().RawPtr()
              + num_remaining_surfel;
        DeviceArrayView<ushort4> appended_surfel_knn(
            appended_knn_ptr, num_appended_surfel);
        m_warpfield_extender->ExtendReferenceNodesAndSE3Sync(
            appended_vertex_view, appended_surfel_knn, m_warp_field);
        // Rebuild the node graph
        m_warp_field->BuildNodeGraph();
        // Update skinning
        if (m_warp_field->CheckAndGetNodeSize() > prev_node_size) {
            m_reference_knn_skinner->UpdateBruteForceSkinningIndexWithNewNodes(
                m_warp_field->ReferenceNodeCoordinates().DeviceArrayReadOnly(),
                prev_node_size);

            // Update skinning
            auto skinner_geometry
                = m_surfel_geometry[fused_geometry_idx]->SkinnerAccess();
            auto skinner_warpfield = m_warp_field->SkinnerAccess();
            m_reference_knn_skinner->PerformSkinningUpdate(
                skinner_geometry, skinner_warpfield, prev_node_size);
        }
        TIME_LOG(TimeLogger::printTimeLog("intergrate"));
    }

    // Unmap attributes
    m_renderer->UnmapFusionMapsFromCuda();
    m_renderer->UnmapSurfelGeometryFromCuda(0);
    m_renderer->UnmapSurfelGeometryFromCuda(1);
    TIME_LOG(TimeLogger::printTimeLog("free render"));

    // Debug save
    if (config.isOfflineRendering() || config.isShowOnline()) {
        const auto with_recent = draw_recent || use_reinit;
        const auto& save_dir
            = createOrGetDataDirectory(m_frame_idx, config.save_path());
        const auto num_fused_vertex
            = m_surfel_geometry[fused_geometry_idx]->NumValidSurfels();
        if (config.isShowOnline()) {
            saveVisualizationMaps(num_fused_vertex, fused_geometry_idx,
                m_camera.GetWorld2CameraEigen(),
                m_camera.GetInitWorld2CameraEigen(), save_dir, with_recent);
        } else {
            saveCameraObservations(observation, save_dir);
            saveSolverMaps(solver_maps, save_dir);
            saveVisualizationMaps(num_fused_vertex, fused_geometry_idx,
                m_camera.GetWorld2CameraEigen(),
                m_camera.GetInitWorld2CameraEigen(), save_dir, with_recent);
        }
        TIME_LOG(TimeLogger::printTimeLog("debug save"));
    }

    // Update the index
    m_frame_idx++;
    m_updated_geometry_index = fused_geometry_idx;
}

/* The method for offline visualization
 */
void surfelwarp::SurfelWarpSerial::saveCameraObservations(
    const surfelwarp::CameraObservation& observation,
    const boost::filesystem::path& save_dir)
{
    auto& config = ConfigParser::Instance();
    // Save the segment mask
    if (config.save_segment_mask) {
        Visualizer::SaveSegmentMask(observation.foreground_mask,
            observation.normalized_rgba_map,
            (save_dir / "foreground_mask.png").string(), 1);
    }
    // Save the clipped, filtered depth image
    if (config.save_filter_depth_image) {
        Visualizer::SaveDepthImage(observation.filter_depth_img,
            (save_dir / "cliped_filter_depth.png").string());
    }
    // Save the raw depth image
    if (config.save_raw_depth_image) {
        Visualizer::SaveDepthImage(
            observation.raw_depth_img, (save_dir / "raw_depth.png").string());
    }
}

void surfelwarp::SurfelWarpSerial::saveCorrespondedCloud(
    const CameraObservation& observation, unsigned vao_idx,
    const boost::filesystem::path& save_dir)
{
    const std::string cloud_1_name = (save_dir / "observation.off").string();
    const std::string cloud_2_name = (save_dir / "model.off").string();

    auto geometry = m_surfel_geometry[vao_idx]->Geometry();
    Visualizer::SaveMatchedCloudPair(observation.vertex_confid_map,
        geometry.live_vertex_confid.ArrayView(),
        m_camera.GetCamera2WorldEigen(), cloud_1_name, cloud_2_name);

    // Also save the reference point cloud
    Visualizer::SavePointCloud(geometry.reference_vertex_confid.ArrayView(),
        (save_dir / "reference.off").string());
}

void surfelwarp::SurfelWarpSerial::saveSolverMaps(
    const surfelwarp::Renderer::SolverMaps& solver_maps,
    const boost::filesystem::path& save_dir)
{
    // Save the rendered albedo maps
    auto& config = ConfigParser::Instance();
    if (config.save_reference_albedo_map)
        Visualizer::SaveNormalizeRGBImage(solver_maps.normalized_rgb_map,
            (save_dir / "rendered_rgb.png").string());

    // Save the index map
    if (config.save_validity_index_map)
        Visualizer::SaveValidIndexMap(solver_maps.index_map, 1,
            (save_dir / "validity_index_map.png").string());

    // Save the computed alignment error map
    if (config.save_alignment_error_map) {
        m_warp_solver->ComputeAlignmentErrorMapDirect();
        cudaTextureObject_t alignment_error_map
            = m_warp_solver->GetAlignmentErrorMap();
        Visualizer::SaveGrayScaleImage(alignment_error_map,
            (save_dir / "alignment_error_map_direct.png").string(), 10.0f);

        // Compute again using nodes
        m_warp_solver->ComputeAlignmentErrorMapFromNode();
        alignment_error_map = m_warp_solver->GetAlignmentErrorMap();
        Visualizer::SaveGrayScaleImage(alignment_error_map,
            (save_dir / "alignment_error_map_node.png").string(), 10.0f);

        // Do statistic about the error value
        const auto node_error = m_warp_solver->GetNodeAlignmentError();
        std::ofstream error_output;
        error_output.open((save_dir / "node_error.txt").string());
        node_error.errorStatistics(error_output);
        error_output.close();
    }
}

void surfelwarp::SurfelWarpSerial::saveVisualizationMaps(
    unsigned int num_vertex, int vao_idx, const Eigen::Matrix4f& world2camera,
    const Eigen::Matrix4f& init_world2camera,
    const boost::filesystem::path& save_dir, bool with_recent)
{
    auto& config = ConfigParser::Instance();
    /* Please make sure when show online only, the following if condition only
     * one can be set to true
     */
    if (config.save_live_normal_map)
        m_renderer->SaveLiveNormalMap(num_vertex, vao_idx, m_frame_idx,
            world2camera, (save_dir / "live_normal.png").string(), with_recent);
    if (config.save_live_albedo_map)
        m_renderer->SaveLiveAlbedoMap(num_vertex, vao_idx, m_frame_idx,
            world2camera, (save_dir / "live_albedo.png").string(), with_recent);
    if (config.save_live_phong_map)
        m_renderer->SaveLivePhongMap(num_vertex, vao_idx, m_frame_idx,
            world2camera, (save_dir / "live_phong.png").string(), with_recent);

    if (config.save_reference_normal_map)
        m_renderer->SaveReferenceNormalMap(num_vertex, vao_idx, m_frame_idx,
            init_world2camera, (save_dir / "reference_normal.png").string(),
            with_recent);
    if (config.save_reference_albedo_map)
        m_renderer->SaveReferenceAlbedoMap(num_vertex, vao_idx, m_frame_idx,
            init_world2camera, (save_dir / "reference_albedo.png").string(),
            with_recent);
    if (config.save_reference_phong_map)
        m_renderer->SaveReferencePhongMap(num_vertex, vao_idx, m_frame_idx,
            init_world2camera, (save_dir / "reference_phong.png").string(),
            with_recent);
}

boost::filesystem::path surfelwarp::SurfelWarpSerial::createOrGetDataDirectory(
    int frame_idx, boost::filesystem::path save_path)
{
    // Construct the path
    std::string frame_path = "/frame_" + std::to_string(frame_idx);
    boost::filesystem::path result_folder = save_path / frame_path;
    if (!boost::filesystem::exists(result_folder)) {
        boost::filesystem::create_directories(result_folder);
    }

    // The directory should be always exist
    return result_folder;
}

// The decision function for integrate and reinit
// Currently not implemented
bool surfelwarp::SurfelWarpSerial::shouldDoIntegration() const
{
    return true;
}

static bool shouldDoReinitConfig(int frame_idx)
{
    using namespace surfelwarp;
    const auto& config = ConfigParser::Instance();

    // Check the config
    // Do not use reinit
    if (!surfelwarp::SurfelWarpSerial::get_use_periodic_reinit()) {
        return false;
    }

    // Check the peroid
    const auto period = config.reinit_period();
    SURFELWARP_CHECK(period > 0);
    if (frame_idx > 0 && (frame_idx % period) == 0) {
        return true;
    } else
        return false;
}

// The processing interface of pipeline
bool surfelwarp::SurfelWarpSerial::shouldDoReinit() const
{
    return shouldDoReinitConfig(m_frame_idx);
}

bool surfelwarp::SurfelWarpSerial::shouldDrawRecentObservation() const
{
    return m_frame_idx - m_reinit_frame_idx
           <= Constants::kStableSurfelConfidenceThreshold + 1;
}
