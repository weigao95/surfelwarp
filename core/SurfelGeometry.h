//
// Created by wei on 3/18/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/ArraySlice.h"
#include "common/DeviceBufferArray.h"
#include "math/device_mat.h"

namespace surfelwarp {
	
	//Forward declaration for class that should have full access to geometry
	struct GLSurfelGeometryVBO;
	class SurfelNodeDeformer;
	class DoubleBufferCompactor;

	class SurfelGeometry {
	private:
		//The underlying struct for the surfel model
		//Read-Write access, but not owned
		DeviceSliceBufferArray<float4> m_reference_vertex_confid;
		DeviceSliceBufferArray<float4> m_reference_normal_radius;
		DeviceSliceBufferArray<float4> m_live_vertex_confid;
		DeviceSliceBufferArray<float4> m_live_normal_radius;
		DeviceSliceBufferArray<float4> m_color_time;
		friend struct GLSurfelGeometryVBO; //map from graphic pipelines
		friend class SurfelNodeDeformer; //deform the vertex/normal given warp field
		friend class DoubleBufferCompactor; //compact from one buffer to another in double buffer setup
		
		//These are owned
		DeviceBufferArray<ushort4> m_surfel_knn;
		DeviceBufferArray<float4> m_surfel_knn_weight;
		
		//The size recorded for recovering
		size_t m_num_valid_surfels;
		
	public:
		using Ptr = std::shared_ptr<SurfelGeometry>;
		SurfelGeometry();
		~SurfelGeometry();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(SurfelGeometry);
		
		//Set the valid size of surfels
		size_t NumValidSurfels() const { return m_num_valid_surfels; }
		void ResizeValidSurfelArrays(size_t size);
		
		//The general read-only access
	public:
		DeviceArrayView<float4> GetReferenceVertexConfidence() const { return m_reference_vertex_confid.ArrayView(); }
		
		/* The read-only accessed by solver
		 */
	public:
		struct SolverInput {
			DeviceArrayView<ushort4> surfel_knn;
			DeviceArrayView<float4> surfel_knn_weight;
		};
		SolverInput SolverAccess() const;

		/* The accessed by legacy solver
		 */
		struct LegacySolverInput {
			DeviceArray<ushort4> surfel_knn;
			DeviceArray<float4> surfel_knn_weight;
		};
		LegacySolverInput LegacySolverAccess();
	
		
		/* The fusion handler will write to these elements, while maintaining
		 * a indicator that which surfel should be compacted to another buffer.
		 */
		struct SurfelFusionInput {
			DeviceArraySlice<float4> live_vertex_confid;
			DeviceArraySlice<float4> live_normal_radius;
			DeviceArraySlice<float4> color_time;
			DeviceArrayView<ushort4> surfel_knn;
			DeviceArrayView<float4> surfel_knn_weight;
		};
		SurfelFusionInput SurfelFusionAccess();
		
		
		/* The reference skinner will read the reference vertex, and overwrite
		 * the knn index and the weights.
		 */
		struct SkinnerInput {
			DeviceArrayView<float4> reference_vertex_confid;
			DeviceArraySlice<ushort4> surfel_knn;
			DeviceArraySlice<float4> surfel_knn_weight;
		};
		SkinnerInput SkinnerAccess();
		
		
		/* The read-only access to memebers
		 */
		DeviceArrayView<float4> ReferenceVertexArray() const { return m_reference_vertex_confid.ArrayView(); }
		DeviceArrayView<ushort4> SurfelKNNArray() const { return m_surfel_knn.ArrayView(); }

		
		/* The method to for debuging
		 */
		void AddSE3ToVertexAndNormalDebug(const mat34& se3);
		
		//Typically for visualization
		struct GeometryAttributes {
			DeviceArraySlice<float4> reference_vertex_confid;
			DeviceArraySlice<float4> reference_normal_radius;
			DeviceArraySlice<float4> live_vertex_confid;
			DeviceArraySlice<float4> live_normal_radius;
			DeviceArraySlice<float4> color_time;
		};
		GeometryAttributes Geometry();
	};
}
