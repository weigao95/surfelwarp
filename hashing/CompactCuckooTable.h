//
// Created by wei on 2/17/18.
//

#pragma once

#include "hashing/hash_config.h"
#include "hashing/hash_common.h"


namespace hashing {

	/**
	 * \brief The class customized for two operations:
	 *        forall(valid voxels) do some thing; and
	 *        (parallel) query the existence and index of a given voxel;
	 */
	class CompactCuckooTable
	{
	private:
	public:
		unsigned m_max_entries;
		/* Buffer for scratch Hash Table
		* */
		KeyT* m_scratch_table;
		unsigned m_table_size;
		HashConstants m_hash_constants;

		//The stash stratch table is just a pointer to m_scratch table
		KeyT* m_stash_scratch_table;
		uint2 m_stash_constants;

		//Book-keeping constants
		unsigned int* m_stash_flag;
		unsigned int* m_failures;

		/* Buffer for compacted keys and the true hash table
		 * The element value in hash table is the index in compacted_key
		* */
		KeyT* m_compacted_keys;
		unsigned m_compacted_key_size;
		HashEntry* m_hash_table; //The same layout as m_scratch_table
		HashEntry* m_stash_table;

		//Temp storage for cub selection
		size_t m_temp_bytes;
		unsigned* m_unique_indicator;
        unsigned* m_prefixsum_unique_indicator;
		unsigned char* m_temp_storage;


	public:
		explicit CompactCuckooTable();
		~CompactCuckooTable();
		CompactCuckooTable(const CompactCuckooTable&) = delete;
		CompactCuckooTable& operator=(const CompactCuckooTable&) = delete;

		//Allocate buffer for given size
		void AllocateBuffer(const unsigned int max_entries, cudaStream_t stream = 0);
		void ReleaseBuffer();

		//Reset the table
		void ResetScratchTable(cudaStream_t stream = 0);

        //The main interface
        bool InsertScratchTable(const KeyT* d_keys, const unsigned num_entries, cudaStream_t stream = 0);
		void CompactKeys(cudaStream_t stream = 0);
        bool Insert(const KeyT* d_keys, const unsigned num_entries, cudaStream_t stream = 0);

		//The interface functions
        unsigned TableSize() const { return m_table_size; }
        unsigned CompactKeySize() const { return m_compacted_key_size; }
        const KeyT* ScratchTable() const { return m_scratch_table; }
        const KeyT* CompactedKeyArray() const { return m_compacted_keys; }

		/*
		 * The encoder and decoder for voxel. 
		 * In this version, assume x, y and z are in the range of [-512, 512]
		 */
		__host__ __device__ __forceinline__ static 
		KeyT encode_voxel(const int x, const int y, const int z)
		{
			return (x + 512) + (y + 512) * 1024 + (z + 512) * 1024 * 1024;
		}

		__host__ __device__ __forceinline__ static 
		void decode_voxel(const KeyT encoded, int& x, int& y, int& z)
		{
			z = encoded / (1024 * 1024);
			x = encoded % 1024;
			y = (encoded - z * 1024 * 1024) / 1024;
			x -= 512;
			y -= 512;
			z -= 512;
		}
	};
}