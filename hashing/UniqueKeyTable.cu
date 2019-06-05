#include "hashing/UniqueKeyTable.h"
#include "hashing/hash_interface.h"
#include <cuda_runtime_api.h>
#include <math.h>

hashing::UniqueKeyTable::UniqueKeyTable() {
    m_max_entries = m_table_size = 0;
    m_table_content = m_stash_table = nullptr;
    m_failures = m_stash_flag = nullptr;
}

hashing::UniqueKeyTable::~UniqueKeyTable() {
    ReleaseBuffer();
}

void hashing::UniqueKeyTable::AllocateBuffer(const int max_entries, cudaStream_t stream) {
	//Already allocate enought buffer
	if (m_max_entries >= max_entries) return;

	//Release the buffer if already hold some
    if(m_table_size > 0) {
        ReleaseBuffer();
    }

    //Determine the table size
    m_max_entries = max_entries;
    m_table_size = ceil(max_entries * space_factor);

    //Allocate the buffer
    unsigned int allocate_size = m_table_size + stash_table_size;
    cudaSafeCall(cudaMalloc((void **) &(m_table_content), allocate_size * sizeof(HashEntry)));
    cudaSafeCall(cudaMalloc((void **) &(m_failures), sizeof(unsigned int)));
    cudaSafeCall(cudaMalloc((void **) &(m_stash_flag), sizeof(unsigned int)));
    m_stash_table = m_table_content + m_table_size;

    //Clear the table element
    ResetTable(stream);
}


void hashing::UniqueKeyTable::ReleaseBuffer() {
    //Release the main table
    m_table_size = 0;
    cudaSafeCall(cudaFree(m_table_content));

    //Release the book-keeping constants
    cudaSafeCall(cudaFree(m_failures));
    cudaSafeCall(cudaFree(m_stash_flag));
}

void hashing::UniqueKeyTable::ResetTable(cudaStream_t stream) {
    //Clear the table entries and flags
    const HashEntry reset_entry = EmptyEntry;
    const unsigned int allocate_size = m_table_size + stash_table_size;

    //Invoke the device functions
    resetEntryArray(allocate_size, reset_entry, m_table_content, stream);
    cudaSafeCall(cudaMemsetAsync(m_failures, 0, sizeof(unsigned int), stream));
    cudaSafeCall(cudaMemsetAsync(m_stash_flag, 0, sizeof(unsigned int), stream));

    //Regenerate the hash and stash constants
    build_hash_constants(m_hash_constants, m_stash_constants);
}

bool hashing::UniqueKeyTable::Insert(const KeyT *d_keys, const unsigned int num_entries, cudaStream_t stream) {
    //Check the size
    if(num_entries > m_max_entries) {
        return false;
    }

    //The loop for insertion
    for (auto i = 0; i < max_restart_attempts; i++) {
        const auto succeed = cuckooInsert(
                num_entries, d_keys,
                m_table_size, m_table_content, m_hash_constants,
                m_stash_table, m_stash_constants, m_stash_flag,
                m_failures, stream
        );
        if(succeed) return true;
        else {
            //Clear the table and reset and constants
            ResetTable(stream);
        }
    }

    //Not succeed after severl attempts
    return false;
}


bool hashing::UniqueKeyTable::Insert(
        const KeyT *d_keys,
        const ValueT *d_values,
        const unsigned int num_entries,
        cudaStream_t stream
) {
    //Check the size
    if(num_entries > m_max_entries) {
        return false;
    }

    //The loop for insertion
    for (auto i = 0; i < max_restart_attempts; i++) {
        const auto succeed = cuckooInsert(
                num_entries, d_keys, d_values,
                m_table_size, m_table_content, m_hash_constants,
                m_stash_table, m_stash_constants, m_stash_flag,
                m_failures, stream
        );
        if(succeed) return true;
        else {
            //Clear the table and reset and constants
            ResetTable(stream);
        }
    }

    //Not succeed after severl attempts
    return false;
}


void hashing::UniqueKeyTable::Retrieve(
        const KeyT *d_keys, ValueT* d_values,
        const unsigned int num_queries,
        cudaStream_t stream
) const {
    retrieveHashValue(
            num_queries, d_keys, d_values,
            m_table_size, m_table_content, m_hash_constants,
            m_stash_table, m_stash_constants, m_stash_flag,
            stream
    );
}