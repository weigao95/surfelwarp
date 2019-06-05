#include "hashing/hash_ops.h"
#include "hashing/hash_config.h"
#include "hashing/hash_interface.h"

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace hashing { namespace device {

	__device__ bool insertEntry(
		HashEntry entry,
		const unsigned int max_attempt_iters,
		HashEntry * table,
		const unsigned int table_size,
		const uint2* hash_constants,
		HashEntry * stash_table,
		const uint2 stash_constants,
		unsigned int * stash_flag
	) {
		KeyT key = entry_key(entry);

		//At first, the key is inserted to the first position
		unsigned int location = hash_value(hash_constants[0], key) % table_size;

		//Attempt insert
		for (int i = 0; i < max_attempt_iters; i++) {
			//Test this one
			entry = atomicExch(&table[location], entry);
			key = entry_key(entry);

			//We found an empty slot, return it
			if (key == EmptyKey) break;

			//Determine the next location
			location = next_location(hash_constants, table_size, location, key);
		}

		//Fail to insert into the main table, insert into the stash table
		if (key != EmptyKey) {
			const auto stash_slot = stash_position(stash_constants, key);
			const HashEntry replaced_entry = atomicCAS(stash_table + stash_slot, EmptyEntry, entry);
			key = entry_key(replaced_entry);
			*stash_flag = 1;
		}

		//If finally this thread evicts an empty slot
		return (key == EmptyKey);
	}

	__device__ ValueT retrieveValue(
		const KeyT key,
		const unsigned int table_size,
		const HashEntry *table_content,
		//The hashing functions
		const uint2* hash_constants,
		//Check the stash
		const HashEntry* stash_table,
		const uint2 stash_constant,
		const unsigned int* stash_flag
	) 
	{
		unsigned int location;
		HashEntry entry = EmptyEntry;

		//First check all the key location
		for (auto i = 0; i < num_hash_funcs; i++) {
			location = hash_value(hash_constants[i], key) % table_size;
			entry = table_content[location];
			if (entry_key(entry) == key)
				break;
		}

		//Next check the stash table
		if (entry_key(entry) != key && (*stash_flag > 0)) {
			const auto stash_slot = stash_position(stash_constant, key);
			entry = stash_table[stash_slot];
		}

		//Return it
		return entry_value(entry);
	}
	
	__global__ void cuckooInsertKernel(
		//The input parameters
		const unsigned int num_entries,
		const KeyT *d_keys,
		const ValueT *d_values,
		//The hash table parameters
		const unsigned int table_size,
		HashEntry *table,
		const HashConstants hash_constants,
		//For stach table
		HashEntry* stash_table,
		const uint2 stash_constant,
		unsigned int *stash_flag,
		//Global constants and flags
		const unsigned int max_attempt_iter,
		unsigned int *failures
	)
	{
		//Obtain the element
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= num_entries || ((*failures) > 0)) return;
		const auto entry = make_entry(d_keys[idx], d_values[idx]);

		//Insert it
		const auto success = insertEntry(
			entry, max_attempt_iter, 
			table, table_size, hash_constants.constants, 
			stash_table, stash_constant, stash_flag
		);

		//Set flag if not success
		if (!success) *failures = 1;
	}


	__global__ void cuckooInsertKernel(
		//The input parameters
		const unsigned int num_entries,
		const KeyT *d_keys,
		//The hash table parameters
		const unsigned int table_size,
		HashEntry *table,
		const HashConstants hash_constants,
		//For stach table
		HashEntry* stash_table,
		const uint2 stash_constant,
		unsigned int *stash_flag,
		//Global constants and flags
		const unsigned int max_attempt_iter,
		unsigned int *failures
	)
	{
		//Obtain the element
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= num_entries || ((*failures) > 0)) return;
		const auto entry = make_entry(d_keys[idx], idx);

		//Insert it
		const auto success = insertEntry(
			entry, max_attempt_iter,
			table, table_size, hash_constants.constants,
			stash_table, stash_constant, stash_flag
		);

		//Set flag if not success
		if (!success) *failures = 1;
	}


	__global__ void setEntryArrayKernel(
		const unsigned int array_size,
		const HashEntry entry,
		HashEntry *entry_array
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < array_size) {
			entry_array[idx] = entry;
		}
	}


	__global__ void retrieveValueKernel(
		const int num_queries,
		const KeyT *d_keys,
		ValueT *d_values,
		//The table parameters
		const unsigned int table_size,
		const HashEntry *table_content,
		const HashConstants hash_constants,
		//The stash table
		const HashEntry* stash_table,
		const uint2 stash_constants,
		const unsigned int* stash_flag
	) {
		//Obtain the key for this query
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= num_queries) return;
		const KeyT key = d_keys[idx];

		//Device function for retrieval
		const auto value = retrieveValue(
			key,
			table_size, table_content, hash_constants.constants, 
			stash_table, stash_constants, stash_flag
		);

		//Store to global memory
		d_values[idx] = value;
	}

}; /* The end of namespace device */ }; /* The end of namespace hashing */




bool hashing::cuckooInsert(
	const unsigned int num_entries, 
	const unsigned int * d_keys, 
	const unsigned int * d_values, 
	const unsigned int table_size,
	HashEntry * d_table_content, 
	const HashConstants & hash_constants, 
	HashEntry* stash_table,
	const uint2 stash_constant, 
	unsigned int * d_stash_flag,
	unsigned int * d_failure,
	cudaStream_t stream
) {
	//Constants for device funcs
	const auto max_attempt_iters = max_insert_attempts(num_entries, table_size);
	unsigned int host_failure = 0;

	//Parameters for kernel invoke
	dim3 blk(insert_thread_block);
	dim3 grid(divUp(num_entries, blk.x));
	device::cuckooInsertKernel<<<grid, blk, 0, stream>>>(
		num_entries, d_keys, d_values,
		table_size, d_table_content, hash_constants,
		stash_table, stash_constant, d_stash_flag,
		max_attempt_iters,
		d_failure
	);

	//Check success or not
	cudaSafeCall(cudaMemcpyAsync(&host_failure, d_failure, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	cudaSafeCall(cudaStreamSynchronize(stream));
	return (host_failure == 0);
}


bool hashing::cuckooInsert(
	const unsigned int num_entries,
	const unsigned int * d_keys,
	const unsigned int table_size,
	HashEntry * d_table_content,
	const HashConstants & hash_constants,
	HashEntry* stash_table,
	const uint2 stash_constant,
	unsigned int * d_stash_flag,
	unsigned int * d_failure,
	cudaStream_t stream
) {
	//Constants for device funcs
	const auto max_attempt_iters = max_insert_attempts(num_entries, table_size);
	unsigned int host_failure = 0;

	//Parameters for kernel invoke
	dim3 blk(insert_thread_block);
	dim3 grid(divUp(num_entries, blk.x));
	device::cuckooInsertKernel<<<grid, blk, 0, stream>>>(
		num_entries, d_keys, 
		table_size, d_table_content, hash_constants,
		stash_table, stash_constant, d_stash_flag,
		max_attempt_iters,
		d_failure
	);

	//Check success or not
	cudaSafeCall(cudaMemcpyAsync(&host_failure, d_failure, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	cudaSafeCall(cudaStreamSynchronize(stream));
	return (host_failure == 0);
}


void hashing::resetEntryArray(
	const unsigned int array_size,
	const HashEntry entry, 
	HashEntry * d_entry_array, 
	cudaStream_t stream
) {
	dim3 blk(256);
	dim3 grid(divUp(array_size, blk.x));
	device::setEntryArrayKernel<<<grid, blk, 0, stream>>>(array_size, entry, d_entry_array);
}

void hashing::retrieveHashValue(
	const int num_queries, 
	const KeyT * d_keys, 
	ValueT * d_values, 
	const unsigned int table_size, 
	const HashEntry * table_content, 
	const HashConstants & hash_constants, 
	const HashEntry * stash_table, 
	const uint2 stash_constants, 
	const unsigned int * d_stash_flag,
	cudaStream_t stream
) {
	dim3 blk(256);
	dim3 grid(divUp(num_queries, blk.x));
	device::retrieveValueKernel<<<grid, blk, 0, stream>>>(
		num_queries, d_keys, d_values,
		table_size, table_content,hash_constants,
		stash_table, stash_constants, d_stash_flag
	);
}