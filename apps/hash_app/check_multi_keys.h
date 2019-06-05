//
// Created by wei on 2/17/18.
//

#pragma once

#include <vector>

#include "hashing/MultiKeyTable.h"


void check_multi_key_building(
        const std::vector<unsigned int>& h_keys,
        const hashing::MultiKeyTable& table
);

void check_multikey_hashing(const size_t test_size = 100000, const unsigned average_duplication = 32);