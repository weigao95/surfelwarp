#pragma once
#include "hashing/TicketBoardSet.h"
#include <cuda_runtime_api.h>
#include <vector>
#include <iostream>

template<typename KeyT>
hashing::TicketBoardSet<KeyT>::TicketBoardSet() 
: m_ticket_board(nullptr), 
  m_table(nullptr),
  m_table_size(0),
  m_compacted_index(nullptr),
  m_temp_storage(nullptr),
  m_valid_indicator(nullptr),
  m_temp_storage_bytes(0)
{
}


template<typename KeyT>
typename hashing::TicketBoardSet<KeyT>::Device hashing::TicketBoardSet<KeyT>::OnDevice() const
{
	Device set;
	set.ticket_board = m_ticket_board;
	set.table = m_table;
	set.table_size = m_table_size;
	set.compacted_index = m_compacted_index;
	set.primary_hash = m_primary_hash;
	set.step_hash = m_step_hash;
	return set;
}

template<typename KeyT>
void hashing::TicketBoardSet<KeyT>::IndexInformation()
{
	//First download the index
	std::vector<unsigned> h_compacted_index;
	h_compacted_index.resize(TableSize());
	cudaSafeCall(cudaMemcpy(
		h_compacted_index.data(), m_compacted_index, 
		sizeof(unsigned) * m_table_size, 
		cudaMemcpyDeviceToHost
	));

	//Do a pass to count the index
	unsigned num_unique_elems = 0;
	for(auto i = 0; i < h_compacted_index.size(); i++) {
		if(h_compacted_index[i] != InvalidIndex) {
			num_unique_elems++;
		}
	}

	//Output the information
	std::cout << "The number of unique elements in this set is " << num_unique_elems << std::endl;
}