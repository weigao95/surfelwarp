//
// Created by wei on 2/17/18.
//

#include "check_multi_keys.h"
#include "common/sanity_check.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <assert.h>
#include <time.h>


void check_multi_key_building(
        const std::vector<unsigned int>& h_keys,
        const hashing::MultiKeyTable& table
) {
    using namespace hashing;

    //Get the size of compaction
    const auto original_key_size = table.OriginalKeySize();
    const auto compacted_key_size = table.CompactedKeySize();
    assert(h_keys.size() == original_key_size);

    //Download the compacted key and value
    std::vector<unsigned> h_compacted_keys;
    std::vector<unsigned> h_compacted_values;
    h_compacted_keys.resize(compacted_key_size);
    h_compacted_values.resize(compacted_key_size);
    cudaMemcpy(h_compacted_keys.data(), table.CompactedKeyArray(), sizeof(unsigned) * compacted_key_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_compacted_values.data(), table.CompactedValueArray(), sizeof(unsigned) * compacted_key_size, cudaMemcpyDeviceToHost);

    //Check them
    for(auto key_idx = 0; key_idx < h_compacted_keys.size(); key_idx++) {
        const auto key = h_compacted_keys[key_idx];
        const auto value = h_compacted_values[key_idx];
        unsigned offset, size;
        MultiKeyTable::decode_value(value, offset, size);
        //Check the size?
        auto h_keys_size = 0;
        for(auto j = 0; j < h_keys.size(); j++) {
            if(h_keys[j] == key) h_keys_size++;
        }
        assert(h_keys_size == size);
    }
}



void check_multikey_hashing(const size_t test_size, const unsigned average_duplication) {
    std::cout << "The checking of multiple keys hash table with size " << test_size << std::endl;
    using namespace hashing;
    using namespace surfelwarp;

    //Prepare the data
    std::vector<unsigned int> h_keys;
    fillMultiKeyValuePairs(h_keys, test_size, 2 * test_size, average_duplication);
    DeviceArray<unsigned int> d_keys;
    d_keys.upload(h_keys);

    //Build the table
    MultiKeyTable table;
    table.AllocateBuffer(h_keys.size());
    assert(table.Insert(d_keys, d_keys.size()));

    //First check the building of compacted key
    check_multi_key_building(h_keys, table);
}

