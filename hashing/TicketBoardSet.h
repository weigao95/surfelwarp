//
// Created by wei on 3/3/18.
//

#pragma once

#include "hashing/hash_common.h"

namespace hashing
{
	/**
	 * \brief The gpu compaction set implementation. Threads
	 *        might insert may duplicate keys, while the set
	 *        only maintain one copy of each key and maps them
	 *        to a continuous indexed region.
	 * \tparam KeyT 
	 */
	template<typename KeyT>
	class TicketBoardSet {
		//The content of the table
		unsigned* m_ticket_board;
		KeyT* m_table;
		unsigned m_table_size;
		unsigned* m_compacted_index;
		uint2 m_primary_hash, m_step_hash;

		//The member interface on host
	public:
		const unsigned* TicketBoard() const { return m_ticket_board; }
		unsigned* TicketBoard() { return m_ticket_board; }
		const KeyT* Table() const { return m_table; }
		KeyT* Table() { return m_table; }
		unsigned TableSize() const { return m_table_size; }
		const uint2& PrimaryHash() const { return m_primary_hash; }
		const uint2& StepHash() const { return m_step_hash; }
		const unsigned* UniqueIndicator() const { return m_valid_indicator; }
		const unsigned* CompactedIndex() const { return m_compacted_index; } 

		//An small struct for interface on device
		struct Device {
			__host__ __device__ Device() {}
			unsigned* ticket_board;
			KeyT* table;
			unsigned* compacted_index;
			unsigned table_size;
			uint2 primary_hash, step_hash;
		};

		//Build a device access struct
		Device OnDevice() const;

		//Temp storeage for compaction
	private:
		unsigned char* m_temp_storage;
		unsigned* m_valid_indicator;
		unsigned m_temp_storage_bytes;

	public:
		//Do not allocate/deallocate buffer
		explicit TicketBoardSet();
		~TicketBoardSet() = default;
		TicketBoardSet(const TicketBoardSet&) = delete;
		TicketBoardSet(TicketBoardSet&&) = delete;
		TicketBoardSet& operator=(const TicketBoardSet&) = delete;
		TicketBoardSet& operator=(TicketBoardSet&&) = delete;

		//Explicit allocate
		void AllocateBuffer(const unsigned max_unique_keys, const float factor = 2);
		void ReleaseBuffer();
		void ResetTicketBoard(cudaStream_t stream = 0);
		void ResetTable(const KeyT empty, cudaStream_t stream = 0);

		//Perform compaction
		void MarkUniqueTickets(cudaStream_t stream = 0);
		void BuildIndex(cudaStream_t stream = 0);

		//Debug methods
		void IndexInformation();
	};
}

//The implementation files
#include "hashing/TicketBoardSet.hpp"
#if defined(__CUDACC__)
#include "hashing/TicketBoardSet.cuh"
#endif