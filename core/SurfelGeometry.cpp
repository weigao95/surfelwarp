#include "core/SurfelGeometry.h"
#include "common/Constants.h"

surfelwarp::SurfelGeometry::SurfelGeometry() : m_num_valid_surfels(0)
{
	//Allocate owned buffer
	m_surfel_knn.AllocateBuffer(Constants::kMaxNumSurfels);
	m_surfel_knn_weight.AllocateBuffer(Constants::kMaxNumSurfels);

	//Resize all arrays to zero
	m_reference_vertex_confid.ResizeArrayOrException(0);
	m_reference_normal_radius.ResizeArrayOrException(0);
	m_live_vertex_confid.ResizeArrayOrException(0);
	m_live_normal_radius.ResizeArrayOrException(0);
	m_color_time.ResizeArrayOrException(0);
	m_surfel_knn.ResizeArrayOrException(0);
	m_surfel_knn_weight.ResizeArrayOrException(0);
}

surfelwarp::SurfelGeometry::~SurfelGeometry()
{
	m_surfel_knn.ReleaseBuffer();
	m_surfel_knn_weight.ReleaseBuffer();
}

void surfelwarp::SurfelGeometry::ResizeValidSurfelArrays(size_t size)
{
	//Resize non-owned geometry
	m_reference_vertex_confid.ResizeArrayOrException(size);
	m_reference_normal_radius.ResizeArrayOrException(size);
	m_live_vertex_confid.ResizeArrayOrException(size);
	m_live_normal_radius.ResizeArrayOrException(size);
	m_color_time.ResizeArrayOrException(size);
			
	//Also resize the owned buffer
	m_surfel_knn.ResizeArrayOrException(size);
	m_surfel_knn_weight.ResizeArrayOrException(size);
			
	//Everything is ok
	m_num_valid_surfels = size;
}


/* The access interface
 */
surfelwarp::SurfelGeometry::SolverInput surfelwarp::SurfelGeometry::SolverAccess() const {
	SolverInput solver_input;
	solver_input.surfel_knn = m_surfel_knn.ArrayReadOnly();
	solver_input.surfel_knn_weight = m_surfel_knn_weight.ArrayReadOnly();
	return solver_input;
}

surfelwarp::SurfelGeometry::LegacySolverInput surfelwarp::SurfelGeometry::LegacySolverAccess() {
	LegacySolverInput solver_input;
	solver_input.surfel_knn = DeviceArray<ushort4>(m_surfel_knn.Ptr(), m_surfel_knn.ArraySize());
	solver_input.surfel_knn_weight = DeviceArray<float4>(m_surfel_knn_weight.Ptr(), m_surfel_knn_weight.ArraySize());
	return solver_input;
}


surfelwarp::SurfelGeometry::SurfelFusionInput surfelwarp::SurfelGeometry::SurfelFusionAccess() {
	SurfelFusionInput fusion_input;
	fusion_input.live_vertex_confid = m_live_vertex_confid.ArraySlice();
	fusion_input.live_normal_radius = m_live_normal_radius.ArraySlice();
	fusion_input.color_time = m_color_time.ArraySlice();
	fusion_input.surfel_knn = m_surfel_knn.ArrayView();
	fusion_input.surfel_knn_weight = m_surfel_knn_weight.ArrayView();
	return fusion_input;
}

surfelwarp::SurfelGeometry::SkinnerInput surfelwarp::SurfelGeometry::SkinnerAccess() {
	SkinnerInput skinner_input;
	skinner_input.reference_vertex_confid = m_reference_vertex_confid.ArrayView();
	skinner_input.surfel_knn = m_surfel_knn.ArraySlice();
	skinner_input.surfel_knn_weight = m_surfel_knn_weight.ArraySlice();
	return skinner_input;
}


/* The debug methods
 */
surfelwarp::SurfelGeometry::GeometryAttributes surfelwarp::SurfelGeometry::Geometry()
{
	GeometryAttributes geometry_attributes;
	geometry_attributes.reference_vertex_confid = m_reference_vertex_confid.ArraySlice();
	geometry_attributes.reference_normal_radius = m_reference_normal_radius.ArraySlice();
	geometry_attributes.live_vertex_confid = m_live_vertex_confid.ArraySlice();
	geometry_attributes.live_normal_radius = m_live_normal_radius.ArraySlice();
	geometry_attributes.color_time = m_color_time.ArraySlice();
	return geometry_attributes;
}