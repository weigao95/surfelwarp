#pragma once

#include "hashing/hash_common.h"

namespace hashing
{
	bool cuckooInsert(
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
		cudaStream_t stream = 0
	);

	bool cuckooInsert(
		const unsigned int num_entries,
		const unsigned int *d_keys,
		const unsigned int table_size,
		HashEntry *d_table_content,
		const HashConstants &hash_constants,
		HashEntry* stash_table,
		const uint2 stash_constant,
		unsigned int *d_stash_flag,
		unsigned int *d_failure,
		cudaStream_t stream = 0
	);

	void resetEntryArray(
		const unsigned int array_size,
		const HashEntry entry,
		HashEntry *d_entry_array,
		cudaStream_t stream = 0
	);

	void retrieveHashValue(
		const int num_queries,
		const KeyT *d_keys,
		ValueT *d_values,
		//The table parameters
		const unsigned int table_size,
		const HashEntry *table_content,
		const HashConstants& hash_constants,
		//The stash table
		const HashEntry* stash_table,
		const uint2 stash_constants,
		const unsigned int* d_stash_flag,
		cudaStream_t stream = 0
	);
}