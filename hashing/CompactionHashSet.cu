#include "hashing/CompactionHashSet.h"
#include "hashing/hash_ops.h"

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cmath>


namespace hashing { namespace device {
	
	__global__ void markValidHashEntryKernel(
		const unsigned* hash_entry,
		const unsigned table_size,
		unsigned* valid_indicator
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < table_size) {
			unsigned valid = 0;
			if(hash_entry[idx] != EmptyKey) {
				valid = 1;
			}
			valid_indicator[idx] = valid;
		}
	}

	__global__ void buildCompactedIndexKernel(
		const unsigned* valid_indicator,
		const unsigned table_size,
		unsigned* compacted_index
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < table_size) {
			unsigned offset = 0xffffffffu;
			if(valid_indicator[idx] > 0) {
				offset = compacted_index[idx] - 1;
			}
			compacted_index[idx] = offset;
		}
	}

}; /* End of namespace device */
}; /* End of namespace hashing */

/* The default contructor
*/
hashing::CompactionHashSet::CompactionHashSet()
	: table(nullptr), compacted_index(nullptr), table_size(0),
	m_temp_storage(nullptr), m_valid_indicator(nullptr), m_temp_storage_bytes(0)
{
}

/* The buffer management method
 */
void hashing::CompactionHashSet::AllocateBuffer(const unsigned max_unique_keys, const float factor)
{
	//The size for main table
	const auto max_table_size = unsigned(std::ceil(factor * max_unique_keys));
	table_size = max_table_size;
	cudaMalloc((void**)(&table), table_size * sizeof(unsigned));
	cudaMalloc((void**)(&compacted_index), table_size * sizeof(unsigned));
	cudaMalloc((void**)(&m_valid_indicator), table_size * sizeof(unsigned));

	//Query the required bytes for temp storage
	size_t required_temp_bytes = 0;
	cub::DeviceScan::InclusiveSum(
		NULL, required_temp_bytes,
		m_valid_indicator, compacted_index,
		table_size
	);
	m_temp_storage_bytes = required_temp_bytes;
	cudaMalloc((void**)&m_temp_storage, m_temp_storage_bytes);

	//Check allocate error
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
}

void hashing::CompactionHashSet::ReleaseBuffer()
{
	cudaFree(table);
	cudaFree(compacted_index);
	cudaFree(m_temp_storage);
	cudaFree(m_valid_indicator);
	//Check free error
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
	table_size = 0;
	m_temp_storage_bytes = 0;
}

void hashing::CompactionHashSet::ResetTable(cudaStream_t stream)
{
	cudaSafeCall(cudaMemsetAsync(table, 0xff, sizeof(unsigned) * table_size, stream));
	build_hash_constants(primary_hash, step_hash);
}


/* The compaction method
 */
void hashing::CompactionHashSet::BuildIndex(cudaStream_t stream)
{
	BuildCompactedHashIndex(
		table, table_size, 
		m_valid_indicator, compacted_index,
		m_temp_storage, m_temp_storage_bytes, 
		stream
	);
}

void hashing::CompactionHashSet::BuildCompactedHashIndex(
	const unsigned * table_entry, unsigned table_size, 
	unsigned * valid_indicator, unsigned* compacted_index,
	unsigned char * temp_storage, unsigned temp_stroage_bytes,
	cudaStream_t stream
) {
	dim3 blk(128);
	dim3 grid(divUp(table_size, blk.x));

	//First mark it
	device::markValidHashEntryKernel<<<grid, blk, 0, stream>>>(
		table_entry, 
		table_size, 
		valid_indicator
	);

	//Do a prefix sum
	size_t required_bytes = temp_stroage_bytes;
	cub::DeviceScan::InclusiveSum(
		(void*)temp_storage, required_bytes, 
		valid_indicator, compacted_index, 
		table_size,
		stream
	);

	//Build the compacted index
	device::buildCompactedIndexKernel<<<grid, blk, 0, stream>>>(
		valid_indicator, 
		table_size, 
		compacted_index
	);
}

void hashing::CompactionHashSet::BuildCompactedIndex(
	const unsigned *valid_indicator, unsigned *compacted_index,
	unsigned table_size,
	unsigned char *temp_storage, unsigned temp_stroage_bytes,
	cudaStream_t stream
) {
	//Do a prefix sum
	size_t required_bytes = temp_stroage_bytes;
	cub::DeviceScan::InclusiveSum(
		(void*)temp_storage, required_bytes,
		valid_indicator, compacted_index,
		table_size,
		stream
	);
	
	//Build the compacted index
	dim3 blk(128);
	dim3 grid(divUp(table_size, blk.x));
	device::buildCompactedIndexKernel<<<grid, blk, 0, stream>>>(
		valid_indicator, table_size, compacted_index
	);
}
