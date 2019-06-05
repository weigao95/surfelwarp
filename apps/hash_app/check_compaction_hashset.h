#pragma once
#include "hashing/CompactionHashSet.h"
#include "hashing/TicketBoardSet.h"
#include "common/common_types.h"

namespace surfelwarp
{
	//Check of compacted unsigned set
	void check_compaction_hashset(const size_t test_size, const size_t test_iters);
	void check_compaction_hashset(const size_t test_size);

	//Check of compacted ticket board set
	void check_permutohedral_set(const size_t test_size, const size_t test_iters);
	void check_permutohedral_set(const size_t test_size);
}