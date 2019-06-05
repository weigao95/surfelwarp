//
// Created by wei on 2/16/18.
//

#pragma once

#include <vector_functions.h>

#include "hashing/hash_common.h"

namespace hashing {


    class MultiKeyTable {
    private:
        /* Buffer for unique valued Hash Table
         * */
        unsigned int m_max_entries;
        unsigned int m_table_buffer_size;
        HashEntry* m_table_content_buffer; //Parameters about hash table buffer allocation

        //Parameter about the single value hash table
        unsigned int m_table_size;
        HashEntry* m_table_content; //Just an alias to m_table_content_buffer
        HashConstants m_hash_constants;

        //Member about stash table
        HashEntry* m_stash_table;
        uint2 m_stash_constants;

        //Book-keeping constants
        unsigned int* m_stash_flag;
        unsigned int* m_failures;


        /* Buffer for sorting and compacting the input keys
         * */
		KeyT* m_sorted_key_buffer;
		unsigned int* m_key_index_buffer;
		unsigned int* m_sorted_index_buffer;
        size_t m_sorted_index_size;
		
		int* m_key_indicator_buffer;
		int* m_prefixsum_indicator_buffer;
		unsigned char* m_temp_storage;
		size_t m_temp_bytes;

		KeyT* m_compacted_key_buffer;
		ValueT* m_compacted_value_buffer;
        size_t m_compacted_key_size;
    public:
        //Constructor and Destrictor
        explicit MultiKeyTable();
        ~MultiKeyTable();
        MultiKeyTable(const MultiKeyTable&) = delete;
        MultiKeyTable& operator=(const MultiKeyTable&) = delete;

        //Allocate buffer for given size
        void AllocateBuffer(const unsigned int max_entries, cudaStream_t stream = 0);
        void ReleaseBuffer();

        //Reset the table
        void ResetTable(cudaStream_t stream = 0);

		//Build the compacted key given the input key
		unsigned int BuildCompactedKeys(const KeyT* d_keys, const unsigned int num_entries, cudaStream_t stream = 0);
        bool Insert(const KeyT* d_keys, const unsigned int num_entries, cudaStream_t stream = 0);

        //The interface function
        unsigned CompactedKeySize() const { return m_compacted_key_size; }
        unsigned OriginalKeySize() const { return m_sorted_index_size; }
        const unsigned* SortedIndexArray() const { return m_sorted_index_buffer; }
        const unsigned* CompactedKeyArray() const { return m_compacted_key_buffer; }
        const unsigned* CompactedValueArray() const { return m_compacted_value_buffer; }

		/*
		 * The method to make value arrays given the input offset. Assume that each key
		 * can be duplicated at most 256 times, thus can be represented in a uint8. 
		 */
		__host__ __device__ __forceinline__ static 
		ValueT make_value(const unsigned int offset, const unsigned int size)
		{
			return ValueT(offset << 8) + size;
		}

		__host__ __device__ __forceinline__ static 
		void decode_value(const ValueT value, unsigned int& offset, unsigned int& size)
		{
			offset = value >> 8;
			size = value & 0x00ff;
		}
    };


}