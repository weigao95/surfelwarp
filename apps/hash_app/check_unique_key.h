//
// Created by wei on 2/16/18.
//

#pragma once

#include <vector>

void fill_unique_kv_pairs(
        std::vector<unsigned int>& h_keys, std::vector<unsigned int>& h_values,
        const unsigned int num_entries, const unsigned int key_maximum
);
void check_uniquekey_hash(const int test_size);