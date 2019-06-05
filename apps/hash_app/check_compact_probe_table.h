#pragma once
#include "check_multi_keys.h"
#include "common/sanity_check.h"
#include "hashing/CompactProbeTable.h"

namespace surfelwarp {

	void check_probe_compaction(const unsigned test_size, const unsigned test_iters);
	void check_probe_compaction(const unsigned test_size);
}
