//
// Created by wei on 2/17/18.
//

#pragma once
#include <vector>
#include "hashing/CompactCuckooTable.h"

void check_scratch_building(
	const std::vector<unsigned int>& h_keys,
	const hashing::CompactCuckooTable& table
);

void check_compaction(
	const std::vector<unsigned int>& h_keys,
	const hashing::CompactCuckooTable& table
);

void check_voxelkey_hashing(const size_t test_size = 100000, const unsigned average_duplication = 32); 
