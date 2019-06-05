#include "custom_hash_compaction.h"
#include "hashing/CompactProbeTable.h"
#include "hashing/hash_ops.h"
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <pcl/common/time.h>
#include <unordered_set>

namespace surfelwarp { namespace device {

	const unsigned max_attempts_probe = 1000;
	const unsigned empty_key = 0xffffffffu;

	__device__ __forceinline__ bool insertEntryProbe(
		const unsigned key,
		unsigned* table,
		const unsigned table_size,
		const uint2 primary_hash, const uint2 step_hash
	) {
		unsigned index = hashing::hash_value(primary_hash, key);
		const unsigned step = hashing::hash_value(step_hash, key) + 1;

		//The main insertation loop
		for (unsigned attempt = 1; attempt <= max_attempts_probe; attempt++) {
			//Make the index inside the table range
			index %= table_size;

			//First check with non-atomic?
			const auto current_elem = table[index];
			if (key == current_elem) return true;
			
			//Already been occupied by another one
			if(current_elem != empty_key) {
				index += attempt * step;
				continue;
			} 
			else { //This is empty key, try to reserve it
				//Atomically check the slot and insert the key if empty
				const auto old_key = atomicCAS(table + index, empty_key, key);

				//Check if the replaced one is empty or the same
				if (old_key == empty_key || old_key == key) {
					return true;
				}

				//Continue probing
				index += attempt * step;
			}
		}

		//After enough probe attempts
		return false;
	}


	//For each key_in, the thread will attempt to insert element
	//from[3 * key_in, 3 * key_in + duplicate)
	__global__ void customizeProbeHashInsert(
		const unsigned* unique_key_in, const unsigned unique_key_size,
		unsigned* table, const unsigned table_size,
		const uint2 primary_hash, const uint2 step_hash,
		unsigned duplicate
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < unique_key_size) {
			const auto key_origin = unique_key_in[idx];
			const auto base_key = key_origin + 3;
			for(auto j = 0; j < duplicate; j++) {
				const auto key = base_key + j;
				insertEntryProbe(key, table, table_size, primary_hash, step_hash);
			}
		
			//Do not check success
			//if (!success) *failure_flag = 1;
		}
	}


	__global__ void customizeCuckooHashInsert(
		const unsigned* unique_key_in, const unsigned unique_key_size,
		unsigned* table, const unsigned table_size,
		const hashing::HashConstants hash_constants,
		//For stach table
		unsigned* stash_scratch_table,
		const uint2 stash_constants,
		//Global constants and flags
		const unsigned max_attempt_iter,
		const unsigned duplicate
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.y;
		if(idx < unique_key_size) {
			const auto key_origin = unique_key_in[idx];
			const auto base_key = key_origin + 3;
			for(auto j = 0; j < duplicate; j++) { //Iterate on different key produced by this once
				auto key = base_key + j;
				auto location = hashing::hash_value(hash_constants.constants[0], key) % table_size;
				unsigned old_key = hashing::EmptyKey;
				for (auto i = 0; i < max_attempt_iter; i++) //Iteraion on attempts
				{
					//Try to insert a new entry
					old_key = key;
					key = atomicExch(&table[location], key);

					// If other key was evicted, we're done.
					if (key == hashing::EmptyKey || key == old_key)
						return;

					//Determine the next location
					location = hashing::next_location(hash_constants.constants, table_size, location, key);
				}

				// Shove it into the stash.
				if (key != hashing::EmptyKey) {
					const auto stash_slot = hashing::stash_position(stash_constants, key);
					key = atomicCAS(stash_scratch_table + stash_slot, hashing::EmptyKey, key);
				}
			}//End loop for duplicate keys
		}//Check for thread in range
	}

}; /* End of namespace device */
}; /* End of namespace surfelwarp */

void surfelwarp::allocateProbeHashBuffer(DeviceArray<unsigned>& unique_keys, hashing::CompactProbeTable & table, const unsigned duplicate)
{
	const auto num_unique_keys_in = unique_keys.size();
	const unsigned num_unique_keys_out = unsigned(1.6 * num_unique_keys_in);
	table.AllocateBuffer(num_unique_keys_out);
	table.ResetTable();
}

void surfelwarp::customizedHashProbeInsert(
	DeviceArray<unsigned>& unique_keys, 
	hashing::CompactProbeTable & table, 
	const unsigned duplicate
) {
	dim3 blk(64);
	dim3 grid(divUp(unique_keys.size(), blk.x));
	device::customizeProbeHashInsert<<<grid, blk>>>(
		unique_keys.ptr(), unique_keys.size(), 
		table.m_table, table.m_table_size, 
		table.m_primary_hash, table.m_step_hash, 
		duplicate
	);
}

void surfelwarp::checkProbeCompactionPerformance(
	DeviceArray<unsigned>& unique_keys, 
	const unsigned test_iters, 
	const unsigned duplicate
) {
	using namespace hashing;
	CompactProbeTable table;
	allocateProbeHashBuffer(unique_keys, table, duplicate);

	//the test session
	{
		pcl::ScopeTime time("Perform compaction using probe table");
		for(auto i = 0; i < test_iters; i++) {
			table.ResetTable();
			customizedHashProbeInsert(
				unique_keys, 
				table, 
				duplicate
			);
		}
		cudaDeviceSynchronize();
	}
	//Check the uniqueness of table
	//Check the table
	std::vector<unsigned> h_entires;
	h_entires.resize(table.TableSize());
	cudaSafeCall(cudaMemcpy(h_entires.data(), table.TableEntry(), sizeof(unsigned) * h_entires.size(), cudaMemcpyDeviceToHost));
	
	//Check the duplicate
	bool unique = isElementUnique(h_entires, 0xffffffff);
	assert(unique);
	//std::cout << "Checking end" << std::endl;
}

void surfelwarp::allocateCuckooTableBuffer(
	DeviceArray<unsigned int> &unique_keys,
	hashing::CompactCuckooTable &table,
	const unsigned duplicate
) {
	const auto num_unique_keys_in = unique_keys.size();
	const unsigned num_unique_keys_out = unsigned(1.7 * num_unique_keys_in);
	table.AllocateBuffer(num_unique_keys_out);
	table.ResetScratchTable();
}


void surfelwarp::customizedCuckooHashInsert(
	DeviceArray<unsigned>& unique_keys, 
	hashing::CompactCuckooTable & table,
	const unsigned duplicate
) {
	const auto max_attempt_iters = hashing::max_insert_attempts(unique_keys.size(), table.m_table_size);
	dim3 blk(64);
	dim3 grid(divUp(unique_keys.size(), blk.x));
	device::customizeCuckooHashInsert<<<grid, blk>>>(
		unique_keys.ptr(), 
		unique_keys.size(), 
		table.m_scratch_table, table.m_table_size, table.m_hash_constants,
		table.m_stash_scratch_table, table.m_stash_constants, 
		max_attempt_iters, 
		duplicate
	);

	//Do compaction
	table.CompactKeys();
}


void surfelwarp::checkCuckooCompactTablePerformance(
	DeviceArray<unsigned>& unique_keys, 
	const unsigned test_iters,
	const unsigned duplicate
) {
	using namespace hashing;
	
	//Construct and allocate the table
	CompactCuckooTable table;
	allocateCuckooTableBuffer(unique_keys, table, duplicate);

	//the test session
	{
		pcl::ScopeTime time("Perform compaction using cuckoo table");
		for (auto i = 0; i < test_iters; i++) {
			table.ResetScratchTable();
			customizedCuckooHashInsert(unique_keys, table, duplicate);
		}
		cudaDeviceSynchronize();
	}

	//std::cout << "Checking end" << std::endl;
}

void surfelwarp::checkCPUCompactionPerformance(
	std::vector<unsigned> &unique_keys,
	const unsigned int test_iters,
	const unsigned int duplicate
) {
	std::unordered_set<unsigned> key_set;
	
	//The checking session
	{
		pcl::ScopeTime time("Perform compaction using cpu set");
		for(auto iter = 0; iter < test_iters; iter++) {
			key_set.clear();
			for(auto duplicate_iter = 0; duplicate_iter < duplicate; duplicate_iter++) {
				for(auto i = 0; i < unique_keys.size(); i++) {
					key_set.insert(unique_keys[i]);
				}//The key iterations
			}//End of duplicate iterations
		}//End of test iterations
	}
}
