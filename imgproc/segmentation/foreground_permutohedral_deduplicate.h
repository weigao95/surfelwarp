#pragma once

#include "common/common_types.h"
#include "imgproc/segmentation/crf_config.h"
#include "imgproc/segmentation/permutohedral_common.h"

namespace surfelwarp {

	struct ForegroundPermutohedralLatticePerHash {
		unsigned short num_lattice;
		LatticeCoordKey<image_permutohedral_dim> lattice_coord_key[max_lattice_per_hash];
		uint2 lattice_coord_offset[max_lattice_per_hash];
	};

	void foregroundDeduplicateHashedLattice(
		const DeviceArray<unsigned>& compacted_hash_offset,
		DeviceArray<unsigned>& sorted_lattice_index, //This might be sorted
		const DeviceArray<LatticeCoordKey<image_permutohedral_dim>>& lattice_coord_key_array,
		DeviceArray<ForegroundPermutohedralLatticePerHash>& compacted_lattice_record,
		unsigned* d_duplicate_flag,
		DeviceArray<unsigned>& deduplicate_reduce_buffer,
		cudaStream_t stream = 0
	);



}