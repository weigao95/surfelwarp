#include "check_compact_probe_table.h"
#include "common/common_types.h"
#include "common/sanity_check.h"
#include "hashing/CompactProbeTable.h"
#include <iostream>

void surfelwarp::check_probe_compaction(const unsigned test_size, const unsigned test_iters)
{
	for(auto i = 0; i < test_iters; i++) {
		check_probe_compaction(test_size);
		if(i % 10 == 0)
			std::cout << "Checking of " << i << std::endl;
	}
}

void surfelwarp::check_probe_compaction(const unsigned test_size)
{
	using namespace hashing;
	using namespace surfelwarp;

	//Prepare data at host
	std::vector<unsigned> h_keys;
	const unsigned duplicate = 32;
	fillMultiKeyValuePairs(h_keys, test_size, 2 * test_size, duplicate);

	//Upload it
	DeviceArray<unsigned> d_keys;
	d_keys.upload(h_keys);

	//Create the table
	CompactProbeTable table;
	table.AllocateBuffer(2 * (h_keys.size() / duplicate));
	table.ResetTable();
	const bool success = table.Insert(d_keys.ptr(), d_keys.size());
	if(!success) {
		std::cout << "Insertation failed!" << std::endl;
	}

	//Check the table
	std::vector<unsigned> h_entires;
	h_entires.resize(table.TableSize());
	cudaSafeCall(cudaMemcpy(h_entires.data(), table.TableEntry(), sizeof(unsigned) * h_entires.size(), cudaMemcpyDeviceToHost));

	//Check the duplicate
	bool unique = isElementUnique(h_entires, 0xffffffff);
	assert(unique);

	//Retrieve of the index
	DeviceArray<unsigned> d_index;
	d_index.create(d_keys.size());
	table.RetrieveIndex(d_keys.ptr(), d_keys.size(), d_index.ptr());

	//Check the retrieved index
	std::vector<unsigned> h_index;
	d_index.download(h_index);
	for(auto i = 0; i < h_index.size(); i++) {
		const auto key_in = h_keys[i];
		assert(h_index[i] < h_entires.size());
		const auto key_retrieved = h_entires[h_index[i]];
		assert(key_in == key_retrieved);
	}
}