#pragma once
#include "common/common_types.h"

namespace surfelwarp{
	

	/**
	 * \brief Given a segmented, subsampled foreground mask, do a 
	 *        conservative upsampling, and perform filtering on it.
	 *        The subsample row/col should be computed using the
	 *        crf_subsample_rate in "segmentation/crf_config.h"
	 */
	void foregroundMaskUpsampleFilter(
		cudaTextureObject_t subsampled_mask,
		unsigned upsample_rows, unsigned upsample_cols,
		float sigma,
		cudaSurfaceObject_t upsampled_mask,
		cudaSurfaceObject_t filter_mask,
		cudaStream_t stream = 0
	);

}
