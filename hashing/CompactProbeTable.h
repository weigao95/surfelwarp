//
// Created by wei on 2/28/18.
//

#pragma once

#include <vector_types.h>
#include <cuda_runtime.h>

namespace hashing {
	
	
	class CompactProbeTable {
	private:
		unsigned* m_table;
		unsigned m_table_size;
		unsigned* m_failure_dev;
		unsigned* m_failure_host;
		uint2 m_primary_hash;
		uint2 m_step_hash;
	
	public:
		//Seems better to allocate buffer explicitly
		explicit CompactProbeTable() = default;
		~CompactProbeTable() = default;

		//Explicit allocate and deallocate the table
		void AllocateBuffer(const unsigned max_unique_key_size);
		void ResetTable(cudaStream_t stream = 0);
		void ReleaseBuffer();

		//The insertation
		bool Insert(const unsigned* d_keys, const unsigned key_size, cudaStream_t stream = 0);

		//Obtain the index
		void RetrieveIndex(const unsigned* d_keys, const unsigned key_size, unsigned* d_index, cudaStream_t stream = 0);

		//Access to members
		const unsigned* TableEntry() const { return m_table; }
		unsigned TableSize() const { return m_table_size; }
	};
	
	
}
