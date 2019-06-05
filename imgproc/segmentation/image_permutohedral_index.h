#pragma once

#include "common/common_types.h"
#include "imgproc/segmentation/crf_config.h"
#include "imgproc/segmentation/permutohedral_common.h"

namespace surfelwarp{
	
	void buildPermutohedralKeys(
		cudaTextureObject_t clip_normalized_rgb_img,
		const float sigma_spatial, const float sigma_rgb,
		const unsigned subsampled_rows, const unsigned subsampled_cols,
		DeviceArray<LatticeCoordKey<image_permutohedral_dim>>& lattice_key,
		DeviceArray<float>& barycentric,
		DeviceArray<unsigned>& lattice_hashed_key,
		DeviceArray<unsigned>& lattice_index,
		cudaStream_t stream = 0
	);


	void markUniqueHashedKey(
		const DeviceArray<unsigned>& sorted_lattice_hashed_key,
		DeviceArray<unsigned>& indicator,
		cudaStream_t stream = 0
	);

	void compactUniqueHashKey(
		const DeviceArray<unsigned>& unique_indicator,
		const DeviceArray<unsigned>& prefixsum_indicator,
		DeviceArray<unsigned>& compacted_offset,
		cudaStream_t stream = 0
	);
}