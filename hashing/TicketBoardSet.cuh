#pragma once
#include "hashing/TicketBoardSet.h"
#include "hashing/CompactionHashSet.h"
#include "hashing/hash_ops.h"
#include "hashing/hash_config.h"
#include "hashing/hash_common.h"
#include "hashing/PrimeTableSizer.h"

#include <cmath>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>


namespace hashing { namespace device {

	template<typename KeyT> __device__ __forceinline__
	bool insertTicketSetEntry(
		const KeyT& key,
		const unsigned hashed_key,
		unsigned* ticket_board,
		KeyT* table,
		const unsigned table_size,
		const uint2& primary_hash, const uint2& step_hash,
		unsigned* flag = nullptr
	) {
		unsigned index = hash_value(primary_hash, hashed_key);
		const unsigned step = hash_value(step_hash, hashed_key) + 1;
		const unsigned key_ticket = hashed_key;

#if defined(COMPACTION_COUNT_BLOCKING)
		unsigned blocked_iters = 0;
#endif

		//The main loop, this version directly use cas
		for(auto attempt = 1; attempt <= max_probe_attempt; attempt++)
		{
			//Make the index inside the table range
			index %= table_size;

			//First check with non-atomic, this read might be stale
			const auto current_ticket = ticket_board[index];
			
			//Already enough to ensure this slot is reserved
			if (current_ticket != key_ticket && current_ticket != EmptyTicket)
			{
				//Someone MUST reserve it, continue probing
				index += attempt * step;
			}
			else if(current_ticket == key_ticket)
			{
				const KeyT written_key = table[index];
				if(written_key == key) {
					//Enough to ensure this is written by same key
					return true;
				}
				
				//written key is not our key
				if(written_key.is_null())
				{
#if defined(COMPACTION_USE_BLOCKING)
#else
					index += attempt * step;
#endif
				}
				else if((!written_key.is_null()) && !(written_key == key))
				{
					//Continue probing as someone else might written
					index += attempt * step;
				}
			}
			else //Try reserve it
			{
				const auto old_ticket = atomicCAS(ticket_board + index, EmptyTicket, key_ticket);
				
				//We have reserve a new ticket
				if(old_ticket == EmptyTicket)
				{
					//Write it
					table[index] = key;
					return true;
				}
				
				//We have not reserve an empty slot
				if(old_ticket != key_ticket)
				{
					//Someone MUST reserve it, continue probing
					index += attempt * step;
				}
				else if(old_ticket == key_ticket)
				{
					const KeyT written_key = table[index];
					if(written_key == key) {
						return true;
					}

					//Someone MIGHT reserve it
					if(written_key.is_null()){
#if defined(COMPACTION_USE_BLOCKING)
						//index += attempt * step;
						//Block here?
						//blocked_iters++;
						//atomicMax(flag, blocked_iters);
#else
						index += attempt * step;
#endif
					} 
					else {
						//Someone MIGHT written
						index += attempt * step;
					}
				}
			}//End of insertation loop
		}

		return false;
	}

	template<typename KeyT> __device__ __forceinline__
	unsigned retrieveTicketSetKeyIndex(
		const KeyT& key,
		const unsigned hashed_key,
		const unsigned* ticket_board,
		const KeyT* table,
		const unsigned table_size,
		const uint2& primary_hash, const uint2& step_hash
	) {
		unsigned index = hash_value(primary_hash, hashed_key);
		const unsigned step = hash_value(step_hash, hashed_key) + 1;

		//The probe loop
		for(auto attempt = 1; attempt <= max_probe_attempt; attempt++)
		{
			//Make the index inside the table range
			index %= table_size;

			//First check the ticket, this read might be stale
			const unsigned current_ticket = ticket_board[index];
			const KeyT current_key = table[index];
			if(current_ticket == hashed_key && current_key == key) {
				//The value match
				return index;
			}

			if(current_ticket == EmptyTicket) {
				//The result is not valid
				return InvalidIndex;
			}

			//Continue probing
			index += attempt * step;
		}
		return InvalidIndex;
	}

	template<typename KeyT> __global__
	void markUniqueTicketKeyKernel(
		unsigned* ticket_board,
		KeyT* table,
		const unsigned table_size,
		const uint2 primary_hash, const uint2 step_hash,
		unsigned* valid_indicator
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < table_size) {
			unsigned valid = 1;
			const unsigned current_ticket = ticket_board[idx];
			const KeyT current_key = table[idx];
			const unsigned hashed_key = current_key.hash();
			if(current_ticket == EmptyTicket) {
				valid = 0;
			} 
			else //Might be valid or duplicate
			{
				const auto first_index = retrieveTicketSetKeyIndex<KeyT>(
					current_key, hashed_key, 
					ticket_board, table, table_size, 
					primary_hash, step_hash
				);
				if(first_index == idx) valid = 1;
				else valid = 0;
			}

			//Write to indicator
			valid_indicator[idx] = valid;
		}
	}

	__global__ void markUniqueTicketKernel(
		const unsigned* ticket_board,
		unsigned table_size,
		unsigned* valid_indicator
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < table_size) {
			unsigned valid = 1;
			if(ticket_board[idx] == EmptyTicket) {
				valid = 0;
			}
			valid_indicator[idx] = valid;
		}
	}

	template<typename KeyT> __global__
	void nullifyTableContentKernel(
		KeyT* table,
		unsigned table_size,
		const KeyT empty_elem
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < table_size) {
			table[idx] = empty_elem;
		}
	}
	
	__device__ __forceinline__ 
	bool insertHashSetUnsignedEntry(
		const unsigned key,
		unsigned* table,
		const unsigned table_size,
		const uint2& primary_hash, const uint2& step_hash
	) {
		unsigned index = hash_value(primary_hash, key);
		const unsigned step = (hash_value(step_hash, key) % table_size) + 1;

		//The main insert loop
		for (unsigned attempt = 1; attempt <= max_probe_attempt; attempt++) {
			//Make the index inside the table range
			index %= table_size;

			//First check with non-atomic, this read might be stale
			const auto current_elem = table[index];
			
			//Already enough to ensure element is inserted
			if (key == current_elem) return true;

			//Already been occupied by another one
			if (current_elem != EmptyKey) {
				index += attempt * step;
				continue;
			}
			else { //This is empty key, try to reserve it
				
				
				//Atomically check the slot and insert the key if empty
				//The returned new_key must be updated, and if it is not
				//Empty key, no one can erease it
				const auto old_key = atomicCAS(table + index, EmptyKey, key);

				//Check if the replaced one is empty or the same
				if (old_key == EmptyKey || old_key == key) {
					return true;
				}

				//Continue probing
				index += attempt * step;
			}
		}

		//After enough probe attempts
		return false;
	}


}; /* End of namespace device */
}; /* End of namespace hashing */

/* The buffer management method
*/
template<typename KeyT>
void hashing::TicketBoardSet<KeyT>::AllocateBuffer(
	const unsigned max_unique_keys, 
	const float factor
) {
	//The size for main table
	auto max_table_size = unsigned(std::ceil(factor * max_unique_keys));
	max_table_size = PrimeTableSizer::GetPrimeTableSize(max_table_size);
	m_table_size = max_table_size;
	cudaMalloc((void**)(&m_table), m_table_size * sizeof(KeyT));
	cudaMalloc((void**)(&m_compacted_index), m_table_size * sizeof(unsigned));
	cudaMalloc((void**)(&m_valid_indicator), m_table_size * sizeof(unsigned));
	cudaMalloc((void**)(&m_ticket_board), m_table_size * sizeof(unsigned));
	

	//Query the required bytes for temp storage
	size_t required_bytes = 0;
	cub::DeviceScan::InclusiveSum(
		NULL, required_bytes,
		m_valid_indicator, m_compacted_index,
		m_table_size
	);
	m_temp_storage_bytes = required_bytes;
	cudaMalloc((void**)&m_temp_storage, m_temp_storage_bytes);

	//Check allocate error
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
}


template<typename KeyT>
void hashing::TicketBoardSet<KeyT>::ReleaseBuffer()
{
	cudaFree(m_ticket_board);
	cudaFree(m_table);
	cudaFree(m_compacted_index);
	cudaFree(m_temp_storage);
	cudaFree(m_valid_indicator);
	//Check free error
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
	m_table_size = 0;
	m_temp_storage_bytes = 0;
}

template<typename KeyT>
void hashing::TicketBoardSet<KeyT>::ResetTicketBoard(cudaStream_t stream)
{
	cudaSafeCall(cudaMemsetAsync(m_ticket_board, 0xff, sizeof(unsigned) * m_table_size, stream));
	build_hash_constants(m_primary_hash, m_step_hash);
}

template <typename KeyT>
void hashing::TicketBoardSet<KeyT>::ResetTable(const KeyT empty, cudaStream_t stream)
{
	//Reset the table content
	dim3 blk(256);
	dim3 grid(divUp(m_table_size, blk.x));
	device::nullifyTableContentKernel<KeyT><<<grid, blk, 0, stream>>>(
		m_table, m_table_size, empty
	);

	//Reset the ticket board
	ResetTicketBoard(stream);
}

template<typename KeyT>
void hashing::TicketBoardSet<KeyT>::MarkUniqueTickets(cudaStream_t stream) {
	dim3 blk(64);
	dim3 grid(divUp(m_table_size, blk.x));
	device::markUniqueTicketKeyKernel<<<grid, blk, 0, stream>>>(
		m_ticket_board, m_table, m_table_size,
		m_primary_hash, m_step_hash,
		m_valid_indicator
	);
}

template<typename KeyT>
void hashing::TicketBoardSet<KeyT>::BuildIndex(cudaStream_t stream) {
	//Require de-deplicate
	MarkUniqueTickets(stream);
	
	//The index building is the same as the one in unsigned set
	CompactionHashSet::BuildCompactedIndex(
		m_valid_indicator, m_compacted_index, m_table_size,
		m_temp_storage, m_temp_storage_bytes,
		stream
	);
}
