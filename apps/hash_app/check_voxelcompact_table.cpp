//
// Created by wei on 2/17/18.
//

#include "check_multi_keys.h"
#include "check_voxelcompact_table.h"
#include "common/sanity_check.h"
#include "hashing/hash_config.h"

#include <iostream>

void check_scratch_building(const std::vector<unsigned int>& h_keys, const hashing::CompactCuckooTable & table)
{
	using namespace hashing;

	//Download the scratch table
	const auto table_size = table.TableSize();
	const unsigned* scratch_table = table.ScratchTable();
	std::vector<unsigned> h_scratch_table;
	h_scratch_table.resize(table_size + stash_table_size);
	cudaSafeCall(cudaMemcpy(h_scratch_table.data(), scratch_table,
                            sizeof(unsigned) * (table_size + stash_table_size), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaDeviceSynchronize());

	//Check that each host key should be contained
	for(auto j = 0; j < h_keys.size(); j++) {
		const auto key = h_keys[j];
		bool found = false;
		for(auto i = 0; i < h_scratch_table.size(); i++) {
			if(key == h_scratch_table[i]) {
				found = true;
				break;
			}
		}
		assert(found);
	}
}

void check_compaction(
    const std::vector<unsigned int>& h_keys,
    const hashing::CompactCuckooTable& table
) {
    using namespace hashing;

    //Download the scratch table
    const auto compact_size = table.CompactKeySize();
    const unsigned* compact_keys = table.CompactedKeyArray();
    std::vector<unsigned> h_compacted_keys;
    h_compacted_keys.resize(compact_size);
    cudaSafeCall(cudaMemcpy(h_compacted_keys.data(), compact_keys,
                            sizeof(unsigned) * (compact_size), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaDeviceSynchronize());

    //Check that each host key should be contained
    for(auto j = 0; j < h_keys.size(); j++) {
        const auto key = h_keys[j];
        bool found = false;
        for(auto i = 0; i < h_compacted_keys.size(); i++) {
            if(key == h_compacted_keys[i]) {
                found = true;
                break;
            }
        }
        assert(found);
    }
}

void check_voxelkey_hashing(const size_t test_size, const unsigned average_duplication)
{
	//Prepare the keys
	std::cout << "The checking of multiple keys hash table with size " << test_size << std::endl;
	using namespace hashing;
	using namespace surfelwarp;

	//Prepare the data
	std::vector<unsigned int> h_keys;
	fillMultiKeyValuePairs(h_keys, test_size, 2 * test_size, average_duplication);
	DeviceArray<unsigned int> d_keys;
	d_keys.upload(h_keys);

	//Build the table
	CompactCuckooTable table;
	table.AllocateBuffer(test_size);
	assert(table.Insert(d_keys, test_size));

	//Check the building of keys
	//check_scratch_building(h_keys, table);
    check_compaction(h_keys, table);
}