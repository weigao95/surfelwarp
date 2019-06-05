//
// Created by wei on 2/16/18.
//

#include <iostream>

#include "check_unique_key.h"
#include "check_multi_keys.h"
#include "check_voxelcompact_table.h"
#include "check_compact_probe_table.h"
#include "check_compaction_hashset.h"

int main() {
	using namespace surfelwarp;
	using namespace hashing;
    std::cout << "Checking of the hash table" << std::endl;
    
	const unsigned test_size = 2 * 100000;
    //check_uniquekey_hash(test_size);
	//check_voxelkey_hashing();
	//check_multikey_hashing();
	//check_probe_compaction(test_size, 1000);
	//check_compaction_hashset(test_size, 1000);
	check_permutohedral_set(test_size, 1000);
	std::cout << "Check end" << std::endl;
}