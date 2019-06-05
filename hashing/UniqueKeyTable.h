//
// Created by wei on 2/16/18.
//

#pragma once

#include "hashing/hash_common.h"
#include "hashing/hash_config.h"

namespace hashing {

    class UniqueKeyTable {
    private:
        //Parameters about hash table
        unsigned int m_max_entries;
        unsigned int m_table_size;
        HashEntry* m_table_content;
        HashConstants m_hash_constants;

        //Member about stash table
        HashEntry* m_stash_table;
        uint2 m_stash_constants;

        //Book-keeping constants
        unsigned int* m_stash_flag;
        unsigned int* m_failures;

    public:
        explicit UniqueKeyTable();
        ~UniqueKeyTable();
        UniqueKeyTable(const UniqueKeyTable&) = delete;
        UniqueKeyTable& operator=(const UniqueKeyTable&) = delete;


        //Allocate memory for given input size
        void AllocateBuffer(const int max_entries, cudaStream_t stream = 0);
        void ReleaseBuffer();

        //Reset the table
        void ResetTable(cudaStream_t stream = 0);

        //Do insertation
        bool Insert(const KeyT* d_keys, const ValueT* d_values, const unsigned int num_entries, cudaStream_t stream = 0);
        bool Insert(const KeyT* d_keys, const unsigned int num_entries, cudaStream_t stream = 0);

        //Get the value from key
        void Retrieve(const KeyT* d_keys, ValueT* d_values, const unsigned int num_entries, cudaStream_t stream = 0) const;
    };
}
