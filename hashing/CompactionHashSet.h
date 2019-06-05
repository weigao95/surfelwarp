#pragma once

#include "hashing/hash_common.h"

namespace hashing
{
	
	/**
	* \brief The struct maintained a map from unsigned
	*        key to compacted index/empty index. The
	*        insertation method is provided as a
	*        skeleton in "hashing/TicketBoardSet.cuh"
	*/
	struct CompactionHashSet {
		unsigned* table;
		unsigned* compacted_index;
		unsigned table_size;
		uint2 primary_hash;
		uint2 step_hash;

		//Temp storeage for compaction
	private:
		unsigned char* m_temp_storage;
		unsigned* m_valid_indicator;
		unsigned m_temp_storage_bytes;

	public:
		explicit CompactionHashSet();
		~CompactionHashSet() = default;
		CompactionHashSet(const CompactionHashSet&) = delete;
		CompactionHashSet(CompactionHashSet&&) = delete;
		CompactionHashSet& operator=(const CompactionHashSet&) = delete;
		CompactionHashSet& operator=(CompactionHashSet&&) = delete;

		//Explicit allocate
		void AllocateBuffer(const unsigned max_unique_keys, const float factor = 2);
		void ReleaseBuffer();
		void ResetTable(cudaStream_t stream = 0);

		//Build the index
		void BuildIndex(cudaStream_t stream = 0);

		//Might be called by other classes
		static void BuildCompactedHashIndex(
			const unsigned* table_entry, unsigned table_size,
			unsigned* valid_indicator, unsigned* compacted_index,
			unsigned char* temp_storage, unsigned temp_stroage_bytes,
			cudaStream_t stream = 0
		);
		static void BuildCompactedIndex(
			const unsigned* valid_indicator, unsigned* compacted_index, unsigned table_size,
			unsigned char* temp_storage, unsigned temp_stroage_bytes,
			cudaStream_t stream = 0
		);
	};

}