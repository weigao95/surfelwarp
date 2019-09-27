//
// Created by wei on 3/29/18.
//

#pragma once

#include "common/Constants.h"
#include "common/ConfigParser.h"
#include "imgproc/ImageProcessor.h"
#include "imgproc/frameio/VolumeDeformFileFetch.h"
#include "core/SurfelGeometry.h"
#include "core/WarpField.h"
#include "core/Camera.h"
#include "core/render/Renderer.h"
#include "core/geometry/LiveGeometryUpdater.h"
#include "core/geometry/GeometryReinitProcessor.h"
#include "core/geometry/GeometryInitializer.h"
#include "core/geometry/KNNBruteForceLiveNodes.h"
#include "core/geometry/ReferenceNodeSkinner.h"
#include "core/geometry/WarpFieldInitializer.h"
#include "core/geometry/WarpFieldExtender.h"
#include "core/warp_solver/WarpSolver.h"
#include "core/warp_solver/RigidSolver.h"

#include <boost/filesystem.hpp>

namespace surfelwarp {
	
	class SurfelWarpSerial {
	private:
		//The primary components
		ImageProcessor::Ptr m_image_processor;
		Renderer::Ptr m_renderer;
		
		//The surfel geometry and their updater
		SurfelGeometry::Ptr m_surfel_geometry[2];
		int m_updated_geometry_index;
		LiveGeometryUpdater::Ptr m_live_geometry_updater;
		
		//The warp field and its updater
		WarpField::Ptr m_warp_field;
		WarpFieldInitializer::Ptr m_warpfield_initializer;
		WarpFieldExtender::Ptr m_warpfield_extender;
		
		//The camera(SE3 transform)
		Camera m_camera;
		
		//The knn index for live and reference nodes
		KNNBruteForceLiveNodes::Ptr m_live_nodes_knn_skinner;
		ReferenceNodeSkinner::Ptr m_reference_knn_skinner;
		
		//The warp solver
		WarpSolver::Ptr m_warp_solver;
		RigidSolver::Ptr m_rigid_solver;
		//::WarpSolver::Ptr m_legacy_solver;
		
		//The component for geometry processing
		GeometryInitializer::Ptr m_geometry_initializer;
		GeometryReinitProcessor::Ptr m_geometry_reinit_processor;
		
		//The frame counter
		int m_frame_idx;
		int m_reinit_frame_idx;
		
	public:
		using Ptr = std::shared_ptr<SurfelWarpSerial>;
		SurfelWarpSerial();
		~SurfelWarpSerial();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(SurfelWarpSerial);
		
		//Process the first frame
		void ProcessFirstFrame();
		void ProcessNextFrameNoReinit();
		void ProcessNextFrameWithReinit(bool offline_save = true);
		//void ProcessNextFrameLegacySolver();
		
		//The testing methods
		void TestGeometryProcessing();
		void TestSolver();
		void TestSolverWithRigidTransform();
		void TestRigidSolver();
		void TestPerformance();

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		
		
		
		/* The method to save the informaiton for offline visualization/debug
		 * Assume the geometry pipeline can be called directly.
		 * These methods should be disabled on Real-Time code
		 */
	private:
		//The camera observation
		void saveCameraObservations(const CameraObservation& observation, const boost::filesystem::path& save_dir);
		
		//The rendered solver maps, required the same cuda context (but not OpenGL context)
		void saveSolverMaps(const Renderer::SolverMaps& solver_maps, const boost::filesystem::path& save_dir);
		
		//Save the coorresponded geometry and obsertion
		void saveCorrespondedCloud(const CameraObservation& observation, unsigned vao_idx, const boost::filesystem::path& save_dir);
		
		
		//The rendered and shaded geometry, This method requires access to OpenGL pipeline
		void saveVisualizationMaps(
			unsigned num_vertex,
			int vao_idx,
			const Eigen::Matrix4f& world2camera,
			const Eigen::Matrix4f& init_world2camera,
			const boost::filesystem::path& save_dir,
			bool with_recent = true
		);
	
		//The directory for this iteration
		static boost::filesystem::path createOrGetDataDirectory(int frame_idx);

		//The decision function for integration and reinit
		bool shouldDoIntegration() const;
		bool shouldDoReinit() const;
		bool shouldDrawRecentObservation() const;
	};
	
}