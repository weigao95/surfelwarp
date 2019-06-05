//
// Created by wei on 2/16/18.
//

#include "check_unique_key.h"
#include "common/sanity_check.h"
#include "hashing/UniqueKeyTable.h"
#include <iostream>
#include <random>
#include <time.h>
#include <assert.h>

void fill_unique_kv_pairs(
        std::vector<unsigned int>& h_keys, std::vector<unsigned int>& h_values,
        const unsigned int num_entries, const unsigned int key_maximum
) {
    assert(num_entries < key_maximum);

    //First fill the kv pairs on host
    h_keys.resize(num_entries);
    h_values.resize(num_entries);

    //The random number generator
    std::default_random_engine generator((unsigned int) time(NULL));
    std::uniform_int_distribution<unsigned int> distribution;

    //Fill the values
    for (auto i = 0; i < num_entries; i++) {
        h_keys[i] = i % key_maximum;
        h_values[i] = distribution(generator);
    }
    //Perform a random shuffle
    std::random_shuffle(h_keys.begin(), h_keys.end());
}

void check_uniquekey_hash(const int test_size) {
    std::cout << "The checking of unique keys hash table with size " << test_size << std::endl;
    using namespace surfelwarp;
    using namespace hashing;

    //Prepare the data for input
    std::vector<unsigned int> h_keys, h_values;
    fill_unique_kv_pairs(h_keys, h_values, test_size, test_size * 2);
    DeviceArray<unsigned int> d_keys, d_values;
    d_keys.upload(h_keys);
    d_values.upload(h_values);

    //Build a hash table
    UniqueKeyTable table;
    table.AllocateBuffer(test_size);
    table.Insert(d_keys, d_values, d_keys.size());

    //Do some random shuffle?
    randomShuffle(h_keys, h_values);
    d_keys.upload(h_keys);

    //Try to retrieve the keys
    DeviceArray<unsigned int> d_values_retrieve;
    d_values_retrieve.create(h_keys.size());
    table.Retrieve(d_keys, d_values_retrieve, d_keys.size());
    std::vector<unsigned int> h_values_dev;
    d_values_retrieve.download(h_values_dev);

    //Check it
    for (unsigned int i = 0; i < h_keys.size(); ++i) {
        assert(h_values[i] == h_values_dev[i]);
    }
}