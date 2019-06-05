#include "common/sanity_check.h"
#include "hashing/CompactProbeTable.h"
#include "hashing/CompactCuckooTable.h"

namespace surfelwarp {
	

	//The customized insertation
	//One thread might deal with multiple elements
	//But they can be compute from the same input
	void allocateProbeHashBuffer(
		DeviceArray<unsigned>& unique_keys,
		hashing::CompactProbeTable& table,
		const unsigned duplicate = 32
	);
	void customizedHashProbeInsert(
		DeviceArray<unsigned>& unique_keys,
		hashing::CompactProbeTable& table,
		const unsigned duplicate = 32
	);
	void checkProbeCompactionPerformance(
		DeviceArray<unsigned>& unique_keys,
		const unsigned test_iters = 1000,
		const unsigned duplicate = 32
	);
	
	
	//For cuckoo type compaction table
	void allocateCuckooTableBuffer(
		DeviceArray<unsigned>& unique_keys,
		hashing::CompactCuckooTable& table,
		const unsigned duplicate = 32
	);
	void customizedCuckooHashInsert(
		DeviceArray<unsigned>& unique_keys,
		hashing::CompactCuckooTable& table,
		const unsigned duplicate = 32
	);
	void checkCuckooCompactTablePerformance(
		DeviceArray<unsigned>& unique_keys,
		const unsigned test_iters = 1000,
		const unsigned duplicate = 32
	);
	
	void checkCPUCompactionPerformance(
		std::vector<unsigned>& unique_keys,
		const unsigned test_iters = 1000,
		const unsigned duplicate = 32
	);
}