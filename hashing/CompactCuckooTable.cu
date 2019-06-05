#include "hashing/CompactCuckooTable.h"
#include "hashing/hash_interface.h"
#include "hashing/hash_config.h"
#include "hashing/hash_ops.h"
#include "hashing/hash_common.h"

#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include <math.h>
#include <device_launch_parameters.h>

namespace hashing { namespace device {
	
	__global__ void cuckooCompactInsertKernel(
		//The input parameters
		const unsigned int num_entries,
		const KeyT *d_keys,
		//The hash table parameters
		const unsigned int table_size,
		KeyT *scratch_table,
		const HashConstants hash_constants,
		//For stach table
		unsigned* stash_scratch_table,
		const uint2 stash_constants,
		unsigned *stash_flag,
		//Global constants and flags
		const unsigned int max_attempt_iter,
		unsigned *failures
	)
	{
		//Obtain the element
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= num_entries || ((*failures) > 0)) return;

		//The key for this thread
		auto key = d_keys[idx];
		auto location = hash_value(hash_constants.constants[0], key) % table_size;

		//Keep insert until we found a empty or a copy
		unsigned old_key = EmptyKey;
		for (auto i = 0; i < max_attempt_iter; i++)
		{
			//Try to insert a new entry
			old_key = key;
			key = atomicExch(&scratch_table[location], key);

			// If other key was evicted, we're done.
			if (key == EmptyKey || key == old_key)
				return;

			//Determine the next location
			location = next_location(hash_constants.constants, table_size, location, key);
		}

		// Shove it into the stash.
		if (key != EmptyKey) {
			const auto stash_slot = stash_position(stash_constants, key);
			key = atomicCAS(stash_scratch_table + stash_slot, EmptyKey, key);
			*stash_flag = 1;
		}

		// Check insertation failure
		if (key != EmptyKey)
		{
			*failures = 1;
		}
	}

	
	__global__ void markDuplicateKeysKernel(
		//Parameters about the hash table
		const unsigned table_size,
		KeyT *scratch_table,
		const HashConstants hash_constants,
		//The stash table parameter
		const uint2 stash_constants,
		//The output indicator
		unsigned* is_unique
	)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < (table_size + stash_table_size))
		{
			//Compute the location
			const auto key = scratch_table[idx];
			const auto location_0 = hash_value(hash_constants.constants[0], key) % table_size;
			const auto location_1 = hash_value(hash_constants.constants[1], key) % table_size;
			const auto location_2 = hash_value(hash_constants.constants[2], key) % table_size;
			const auto location_3 = hash_value(hash_constants.constants[3], key) % table_size;
			const auto stash_loc = table_size + stash_position(stash_constants, key);;

			//Figure out where the key is first located.
			unsigned first_index;
			if (scratch_table[location_0] == key) first_index = location_0;
			else if (scratch_table[location_1] == key) first_index = location_1;
			else if (scratch_table[location_2] == key) first_index = location_2;
			else if (scratch_table[location_3] == key) first_index = location_3;
			else                              first_index = stash_loc;

			//If this thread got a later copy of the key, remove this thread's copy
			//from the table.
			if (first_index != idx || key == EmptyKey) {
				scratch_table[idx] = EmptyKey;
				is_unique[idx] = 0;
			}
			else {
				is_unique[idx] = 1;
			}
		}
	}


	__global__ void compactTableKernel(
		const KeyT* scratch_table,
		const unsigned* prefixsum_indicator_array,
		const unsigned table_size,
		HashEntry* hash_table,
		KeyT* compacted_key
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < (table_size + stash_table_size))
		{
			const auto key = scratch_table[idx];
			const auto index = prefixsum_indicator_array[idx] - 1;
			const auto entry = make_entry(key, index);
			//First store the table entry
			hash_table[idx] = entry;
			//If this key is valid, compact it
			if (key != EmptyKey) 
				compacted_key[index] = key;
		}
	}


}; /* End of namespace device */ }; /* End of namespace hashing */

hashing::CompactCuckooTable::CompactCuckooTable()
{
	//Zero-intialize the buffer for scartch table
	m_max_entries = m_table_size = 0;
	m_scratch_table = m_stash_scratch_table = nullptr;
	m_stash_flag = m_failures = nullptr;

	//Zero-initalize the buffer for compacted key and hash table
	m_compacted_key_size = 0;
	m_compacted_keys = nullptr;
	m_hash_table = nullptr;
	m_stash_table = nullptr;
	m_temp_bytes = 0;
	m_temp_storage = nullptr;
	m_unique_indicator = nullptr;
    m_prefixsum_unique_indicator = nullptr;
}

hashing::CompactCuckooTable::~CompactCuckooTable()
{
	ReleaseBuffer();
}

void hashing::CompactCuckooTable::AllocateBuffer(const unsigned max_entries, cudaStream_t stream)
{
	//Already allocate enought buffer
	if (m_max_entries >= max_entries) return;

	//Release the buffer if already hold some
	if (m_table_size > 0) {
		ReleaseBuffer();
	}

	//Determine the table size
	m_max_entries = max_entries;
	m_table_size = ceil(max_entries * space_factor);

	//Allocate the buffer for stratch table
	const auto allocate_size = m_table_size + stash_table_size;
	cudaSafeCall(cudaMalloc((void **) &(m_scratch_table), allocate_size * sizeof(KeyT)));
	cudaSafeCall(cudaMalloc((void **) &(m_failures), sizeof(unsigned int)));
	cudaSafeCall(cudaMalloc((void **) &(m_stash_flag), sizeof(unsigned int)));
	m_stash_scratch_table = m_scratch_table + m_table_size;

	//Allocate the buffer for true hash table
	cudaSafeCall(cudaMalloc((void **) &(m_hash_table), allocate_size * sizeof(HashEntry)));
	cudaSafeCall(cudaMalloc((void **) &(m_unique_indicator), allocate_size * sizeof(unsigned)));
    cudaSafeCall(cudaMalloc((void **) &(m_prefixsum_unique_indicator), allocate_size * sizeof(unsigned)));
	cudaSafeCall(cudaMalloc((void **) &(m_compacted_keys), max_entries * sizeof(KeyT)));
	m_stash_table = m_hash_table + m_table_size;

	//Query the size for prefix-sum
	cub::DeviceScan::InclusiveSum(
		NULL, m_temp_bytes,
        m_unique_indicator, m_prefixsum_unique_indicator,
        allocate_size, stream
    );
	cudaSafeCall(cudaMalloc((void **) &(m_temp_storage), m_temp_bytes));

	//Reset the scratch table
	ResetScratchTable(stream);
}

void hashing::CompactCuckooTable::ReleaseBuffer()
{
	//Clear the size array
	m_max_entries = 0;
	m_table_size = 0;
	m_compacted_key_size = 0;
	m_temp_bytes = 0;

	//Release all buffer
	cudaFree(m_scratch_table);
	cudaFree(m_stash_flag);
	cudaFree(m_failures);
	cudaFree(m_compacted_keys);
	cudaFree(m_hash_table);
	cudaFree(m_unique_indicator);
	cudaFree(m_temp_storage);

	//Sync and check error
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
}

void hashing::CompactCuckooTable::ResetScratchTable(cudaStream_t stream)
{
	const auto allocate_size = m_table_size + stash_table_size;
	cudaSafeCall(cudaMemsetAsync(m_scratch_table, 0xff, sizeof(KeyT) * allocate_size, stream));
	cudaSafeCall(cudaMemsetAsync(m_failures, 0, sizeof(unsigned int), stream));
	cudaSafeCall(cudaMemsetAsync(m_stash_flag, 0, sizeof(unsigned int), stream));

	//Regenerate the hash and stash constants
	build_hash_constants(m_hash_constants, m_stash_constants);
}

bool hashing::CompactCuckooTable::InsertScratchTable(
        const KeyT *d_keys,
        const unsigned num_entries,
        cudaStream_t stream
) {
    //Prepare host variables
    const auto max_attempt_iters = max_insert_attempts(num_entries, m_table_size);
    unsigned int host_failure = 0;
    dim3 insert_blk(insert_thread_block);
    dim3 insert_grid(divUp(num_entries, insert_blk.x));

    //The attempt loop
    for(auto i = 0; i < max_restart_attempts; i++) {
        //Invoke the device function
        device::cuckooCompactInsertKernel<<<insert_grid, insert_blk, 0, stream>>>(
			num_entries, d_keys,
            m_table_size, m_scratch_table, m_hash_constants,
            m_stash_scratch_table, m_stash_constants, m_stash_flag,
            max_attempt_iters, m_failures
        );

        //Check success or not
        cudaSafeCall(cudaMemcpyAsync(&host_failure, m_failures, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
        cudaSafeCall(cudaStreamSynchronize(stream));
        if(host_failure == 0) return true;

        //Re-try
        host_failure = 0;
        ResetScratchTable(stream);
    }

    //Can not find a feasible solution
    return false;
}


void hashing::CompactCuckooTable::CompactKeys(cudaStream_t stream) {
    dim3 compact_blk(256);
    dim3 compact_grid(divUp(m_table_size + stash_table_size, compact_blk.x));
    //First mark the duplicate key
    device::markDuplicateKeysKernel<<<compact_grid, compact_blk, 0, stream>>>(
		m_table_size,
        m_scratch_table,
        m_hash_constants,
        m_stash_constants,
        m_unique_indicator
    );

    //Do a prefix sum
    size_t temp_bytes = m_temp_bytes;
    cub::DeviceScan::InclusiveSum(
        m_temp_storage, temp_bytes,
        m_unique_indicator, m_prefixsum_unique_indicator,
        (m_table_size + stash_table_size), stream
    );

    //Obtain the size of compacted key
    unsigned* d_size_location = m_prefixsum_unique_indicator + (m_table_size + stash_table_size - 1);
    cudaSafeCall(cudaMemcpyAsync(&m_compacted_key_size, d_size_location, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    cudaSafeCall(cudaStreamSynchronize(stream));

	//Do compaction to obtain the final table
	device::compactTableKernel<<<compact_grid, compact_blk, 0, stream>>>(
		m_scratch_table, 
		m_prefixsum_unique_indicator,
		m_table_size, 
		m_hash_table, 
		m_compacted_keys
	);
}


bool hashing::CompactCuckooTable::Insert(const KeyT *d_keys, const unsigned num_entries, cudaStream_t stream) {
    //Frist build a scratch table, only this step may fail
    bool success = InsertScratchTable(d_keys, num_entries, stream);
    if(!success) return false;

    //Compact it
    CompactKeys(stream);
    return true;
}

