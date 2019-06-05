//
// Created by wei on 3/2/18.
//

#pragma once

#include "common/common_types.h"
#include "common/algorithm_types.h"

namespace surfelwarp {
	
	void allocateSortCompactBuffer(size_t input_size, KeyValueSort<unsigned , unsigned>& sort, PrefixSum& prefixSum);
	
	DeviceArray<unsigned> sortCompact(
		const DeviceArray<unsigned>& array_in,
		KeyValueSort<unsigned , unsigned>& sorter,
		DeviceArray<unsigned>& valid_indicator,
		PrefixSum& prefixsum,
		DeviceArray<unsigned>& compacted_buffer
	);

	void checkSortCompactionPerformance(const DeviceArray<unsigned>& array_in, const size_t test_iters = 1000);
}
