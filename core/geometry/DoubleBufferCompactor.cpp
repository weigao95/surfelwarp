//
// Created by wei on 5/7/18.
//

#include "common/Constants.h"
#include "common/ConfigParser.h"
#include "core/geometry/DoubleBufferCompactor.h"

surfelwarp::DoubleBufferCompactor::DoubleBufferCompactor() {
	m_compact_to_geometry = nullptr;
	
	//The row and column of the image
	const auto& config = ConfigParser::Instance();
	m_image_rows = config.clip_image_rows();
	m_image_cols = config.clip_image_cols();
}

surfelwarp::DoubleBufferCompactor::~DoubleBufferCompactor()
{
}

void surfelwarp::DoubleBufferCompactor::SetFusionInputs(
	const RemainingLiveSurfelKNN & remaining_surfels,
	const AppendedObservationSurfelKNN & appended_surfels,
	SurfelGeometry::Ptr compacted_geometry
) {
	m_appended_surfel_knn = appended_surfels;
	m_remaining_surfel = remaining_surfels.live_geometry;
	m_remaining_knn = remaining_surfels.remaining_knn;
	m_compact_to_geometry = compacted_geometry;
}

void surfelwarp::DoubleBufferCompactor::SetReinitInputs(
	const surfelwarp::RemainingLiveSurfel &remaining_surfels,
	const surfelwarp::ReinitAppendedObservationSurfel &append_surfels,
	surfelwarp::SurfelGeometry::Ptr compact_to_geometry
) {
	m_reinit_append_surfel = append_surfels;
	m_remaining_surfel = remaining_surfels;
	m_compact_to_geometry = compact_to_geometry;
}

