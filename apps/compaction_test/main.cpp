//
// Created by wei on 3/2/18.
//

#include <iostream>
#include "hashing/CompactCuckooTable.h"
#include "hashing/CompactProbeTable.h"
#include "common/common_types.h"
#include "sort_compaction.h"
#include "common/sanity_check.h"
#include "custom_hash_compaction.h"

int main() {
	using namespace surfelwarp;
	using namespace hashing;
	std::cout << "The performance test of compaction" << std::endl;
	
	//Prepare data
	const unsigned duplicate = 32;
	const unsigned num_unique_key = 100000;
	const unsigned num_total_keys = duplicate * num_unique_key;
	std::vector<unsigned> h_array_in;
	fillMultiKeyValuePairs(h_array_in, num_total_keys, 2 * num_total_keys, duplicate);
	DeviceArray<unsigned> d_array_in;
	d_array_in.upload(h_array_in);

	//First try sorting
	const unsigned test_iters = 100;
	checkSortCompactionPerformance(d_array_in, test_iters);

	//Next for probing
	//Prepare the data
	h_array_in.clear();
	d_array_in.release();
	for(auto i = 0; i < num_unique_key; i++) {
		h_array_in.push_back(i);
	}

	//A random permutation
	randomShuffle(h_array_in);
	d_array_in.upload(h_array_in);
	
	//Check it
	checkProbeCompactionPerformance(d_array_in, test_iters, duplicate);

	//Check of cuckoo table
	checkCuckooCompactTablePerformance(d_array_in, test_iters, duplicate);
	
	//The cpu version
	checkCPUCompactionPerformance(h_array_in, test_iters, duplicate);
}
