#include "hashing/MultiKeyTable.h"
#include "hashing/hash_common.h"
#include "hashing/hash_ops.h"
#include "hashing/hash_interface.h"

#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>

namespace hashing { namespace device {
	
	__global__ void buildKeyIndexKernel(
		unsigned int* key_index_array,
		const unsigned int key_size 
	)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < key_size) key_index_array[idx] = idx;
	}


	__global__ void markSortedKeyKernel(
		const KeyT* sorted_key, const unsigned int key_size,
		int* indicator_array
	)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		int mark_value = 0;
		if (idx == 0) mark_value = 1;
		else if (idx < key_size && (sorted_key[idx] != sorted_key[idx - 1])) mark_value = 1;
		//Store to global memory
		if(idx < key_size) indicator_array[idx] = mark_value;
	}


	/**
	 * \brief 
	 * \param indicator_array In the size of original_key_size
	 * \param prefixsum_indicator_array The inclusive sum of indicator_array, also in the size of original_key_size
	 * \param sorted_key 
	 * \param original_key_size 
	 * \param compacted_key In the size of prefixsum_indicator[original_key_size - 1]
	 * \param compacted_offset In the size of prefixsum_indicator[original_key_size - 1] + 1,
	 *                         [compacted_offset[i], compacted_offset[i + 1]) is the range of sorted_key[i]
	 */
	__global__ void compactSortedKeyKernel(
		const int* indicator_array,
		const int* prefixsum_indicator_array,
		const KeyT* sorted_key,
		const unsigned int original_key_size,
		KeyT* compacted_key,
		unsigned int* compacted_offset
	)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < original_key_size && indicator_array[idx] == 1)
		{
			const auto compacted_idx = prefixsum_indicator_array[idx] - 1;
			compacted_key[compacted_idx] = sorted_key[idx];
			compacted_offset[compacted_idx] = idx;
		}
		if(idx == original_key_size)
		{
			compacted_offset[prefixsum_indicator_array[original_key_size - 1]] = original_key_size;
		}
	}

	__global__ void buildValueKernel(
		const unsigned int* compacted_offset, 
		const unsigned int compacted_key_size,
		ValueT* value_array
	)
	{
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < compacted_key_size)
		{
			const auto offset = compacted_offset[idx];
			const auto size = compacted_offset[idx + 1] - offset;
			const ValueT value = MultiKeyTable::make_value(offset, size);
			value_array[idx] = value;
		}
	}

}; /* End of namespace device */ }; /* End of namespace hashing */


hashing::MultiKeyTable::MultiKeyTable() {
    //Zero-initialize the variable related to single key table
    m_max_entries = m_table_buffer_size = m_table_size = 0;
    m_table_content_buffer = m_stash_table = m_table_content = nullptr;
    m_failures = m_stash_flag = nullptr;

    //Zero-initialize the buffer to sort the keys
	m_sorted_index_buffer = m_key_index_buffer = m_sorted_index_buffer = m_sorted_key_buffer = nullptr;
	m_key_indicator_buffer = m_prefixsum_indicator_buffer = nullptr;
	m_temp_storage = nullptr;
	m_temp_bytes = m_compacted_key_size = 0;
}


hashing::MultiKeyTable::~MultiKeyTable() {
    ReleaseBuffer();
}


void hashing::MultiKeyTable::AllocateBuffer(const unsigned int max_entries, cudaStream_t stream) {
	//Already allocate enought buffer
	if (m_max_entries >= max_entries) return;

	//Clear the buffer if already hold some
    if(m_table_buffer_size > 0) {
        ReleaseBuffer();
    }

    //Determine the table size
    m_max_entries = max_entries;
    m_table_buffer_size = ceil(max_entries * space_factor);

    //Allocate buffer for single key table
    const auto allocate_size = m_table_buffer_size + stash_table_size;
    cudaSafeCall(cudaMalloc((void **) (&m_table_content_buffer), allocate_size * sizeof(HashEntry)));
    cudaSafeCall(cudaMalloc((void **) (&m_failures), sizeof(unsigned int)));
    cudaSafeCall(cudaMalloc((void **) (&m_stash_flag), sizeof(unsigned int)));
    m_stash_table = m_table_content_buffer + m_table_buffer_size;
    m_table_content = m_table_content_buffer;

    //Allocate the buffer for sorting and compacting
	cudaMalloc((void**)(&m_sorted_key_buffer), max_entries * sizeof(unsigned int));
	cudaMalloc((void**)(&m_key_index_buffer), max_entries * sizeof(unsigned int));
	cudaMalloc((void**)(&m_sorted_index_buffer), max_entries * sizeof(unsigned int));
	cudaMalloc((void**)(&m_key_indicator_buffer), max_entries * sizeof(unsigned int));
	cudaMalloc((void**)(&m_prefixsum_indicator_buffer), max_entries * sizeof(unsigned int));

	//Query the size for sorting and allocate them
	cub::DeviceRadixSort::SortPairs(
		NULL, m_temp_bytes,
        m_sorted_key_buffer, m_sorted_key_buffer,
		m_key_index_buffer, m_sorted_index_buffer,
		max_entries, 0, 32
	);
	cudaMalloc((void**)(&m_temp_storage), m_temp_bytes);

	//Allocate the compacted buffer
	cudaMalloc((void**)(&m_compacted_key_buffer), max_entries * sizeof(unsigned int));
	cudaMalloc((void**)(&m_compacted_value_buffer), max_entries * sizeof(unsigned int));
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());

    //Reset the table
    ResetTable(stream);
}


void hashing::MultiKeyTable::ReleaseBuffer() {
    //Release the buffer related to single key hash table
    m_table_size = 0;
    m_table_buffer_size = 0;
    cudaSafeCall(cudaFree(m_table_content_buffer));

    //Release the book-keeping constants
    cudaSafeCall(cudaFree(m_failures));
    cudaSafeCall(cudaFree(m_stash_flag));

	//Release the buffer for sorting and compacting value
	cudaFree(m_sorted_key_buffer);
	cudaFree(m_key_index_buffer);
	cudaFree(m_sorted_index_buffer);

	cudaFree(m_key_indicator_buffer);
	cudaFree(m_prefixsum_indicator_buffer);
	cudaFree(m_temp_storage);

	cudaFree(m_compacted_key_buffer);
	cudaFree(m_compacted_value_buffer);
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
}

void hashing::MultiKeyTable::ResetTable(cudaStream_t stream) {
	//Clear the table entries and flags
	const HashEntry reset_entry = EmptyEntry;
	const unsigned int allocate_size = m_table_buffer_size + stash_table_size;

	//Invoke the device functions
	resetEntryArray(allocate_size, reset_entry, m_table_content, stream);
	cudaSafeCall(cudaMemsetAsync(m_failures, 0, sizeof(unsigned int), stream));
	cudaSafeCall(cudaMemsetAsync(m_stash_flag, 0, sizeof(unsigned int), stream));

	//Regenerate the hash and stash constants
	build_hash_constants(m_hash_constants, m_stash_constants);
}

unsigned int hashing::MultiKeyTable::BuildCompactedKeys(const KeyT * d_keys, const unsigned int num_entries, cudaStream_t stream)
{
	//Build the key index array
	dim3 build_key_blk(256);
	dim3 build_key_grid(divUp(num_entries, build_key_blk.x));
	device::buildKeyIndexKernel<<<build_key_grid, build_key_blk, 0, stream>>>(m_key_index_buffer, num_entries);

	//Sort the key-index array
	size_t temp_bytes = m_temp_bytes;
	cub::DeviceRadixSort::SortPairs(
		m_temp_storage, temp_bytes,
		d_keys, m_sorted_key_buffer,
		m_key_index_buffer, m_sorted_index_buffer,
		num_entries, 0, 32, stream
	);
    m_sorted_index_size = num_entries;

	//Label the sorted key
	dim3 label_blk(256);
	dim3 label_grid(divUp(num_entries, label_blk.x));
	device::markSortedKeyKernel<<<label_grid, label_blk, 0, stream>>>(
		m_sorted_key_buffer, 
		num_entries, 
		m_key_indicator_buffer
	);

	//Do prefixsum on the label
	cub::DeviceScan::InclusiveSum(
		m_temp_storage, temp_bytes,
        m_key_indicator_buffer, m_prefixsum_indicator_buffer,
		num_entries, stream
	);

    //Query the compacted size
    int compacted_key_size = 0;
    int* d_size_location = m_prefixsum_indicator_buffer + (num_entries - 1);
    cudaSafeCall(cudaMemcpyAsync(&(compacted_key_size), d_size_location, sizeof(int), cudaMemcpyDeviceToHost, stream));
	cudaSafeCall(cudaStreamSynchronize(stream));
	
    //Compact the prefixsum label
	dim3 compact_blk(256);
	dim3 compact_grid(divUp(num_entries, compact_blk.x));
	device::compactSortedKeyKernel<<<compact_grid, compact_blk, 0, stream>>>(
		m_key_indicator_buffer,
		m_prefixsum_indicator_buffer, 
		m_sorted_key_buffer,
		num_entries, 
		m_compacted_key_buffer, 
		m_key_index_buffer
	);


	//Build the value array
	dim3 value_blk(256);
	dim3 value_grid(divUp(compacted_key_size, value_blk.x));
	device::buildValueKernel<<<value_grid, value_blk, 0, stream>>>(
		m_key_index_buffer, 
		compacted_key_size,
		m_compacted_value_buffer
	);

	//Return the size of compacted key
    m_compacted_key_size = compacted_key_size;
	return compacted_key_size;
}

bool hashing::MultiKeyTable::Insert(
        const KeyT *d_keys,
        const unsigned int num_entries,
        cudaStream_t stream
) {
	//First build the compacted key
	const auto compacted_key_size = BuildCompactedKeys(d_keys, num_entries, stream);

	//Build the hash table on the compacted key
	for (auto i = 0; i < max_restart_attempts; i++) {
		m_table_size = ceil(m_compacted_key_size * space_factor);
		const auto succeed = cuckooInsert(
			compacted_key_size, m_compacted_key_buffer, m_compacted_value_buffer,
			m_table_size, m_table_content, m_hash_constants,
			m_stash_table, m_stash_constants, m_stash_flag,
			m_failures, stream
		);
		if (succeed) return true;
		else {
			//Clear the table and reset and constants
			ResetTable(stream);
		}
	}

	//Not succeed after severl attempts
	return false;
}