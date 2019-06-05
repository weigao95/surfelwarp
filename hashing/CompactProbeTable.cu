#include "hashing/CompactProbeTable.h"
#include "hashing/hash_ops.h"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

namespace hashing { namespace device {

	const unsigned max_attempts_probe = 1000;
	const unsigned empty_key = 0xffffffffu;

	__device__ __forceinline__ bool insertEntryHashProbe(
		const unsigned key,
		unsigned* table,
		const unsigned table_size,
		const uint2 primary_hash, const uint2 step_hash
	) {
		unsigned index = hash_value(primary_hash, key);
		const unsigned step = hash_value(step_hash, key) + 1;

		//The main insertation loop
		for(unsigned attempt = 1; attempt <= max_attempts_probe; attempt++) {
			//Make the index inside the table range
			index %= table_size;

			//Atomically check the slot and insert the key if empty
			const auto old_key = atomicCAS(table + index, empty_key, key);

			//Check if the replaced one is empty or the same
			if(old_key == empty_key || old_key == key) {
				return true;
			}

			//Continue probing
			index += attempt * step;
		}

		//After enough probe attempts
		return false;
	}


	__device__ __forceinline__ unsigned retrieveEntryIndexProbeHash(
		const unsigned key,
		const unsigned* table,
		const unsigned table_size,
		const uint2 primary_hash, const uint2 step_hash
	) {
		unsigned index = hash_value(primary_hash, key);
		const unsigned step = hash_value(step_hash, key) + 1;

		//The main search loop
		for (unsigned attempt = 1; attempt <= max_attempts_probe; attempt++) {
			//Make the index inside the table range
			index %= table_size;

			//Read this key
			const auto table_key = table[index];

			//Check if the replaced one is empty or the same
			if (table_key == key) {
				return index;
			} 
			
			//Do not need continue
			if(table_key == empty_key) {
				return 0xffffffff;
			}

			//Continue probing
			index += attempt * step;
		}

		//After enough probe attempts
		return 0xffffffff;
	}

	__global__ void compactProbeInsertEntryKernel(
		const unsigned* key_in, const unsigned key_size,
		unsigned* table, const unsigned table_size,
		const uint2 primary_hash, const uint2 step_hash,
		unsigned* failure_flag
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < key_size) {
			const auto key = key_in[idx];
			const bool success = insertEntryHashProbe(key, table, table_size, primary_hash, step_hash);
			if (!success) *failure_flag = 1;
		}
	}

	__global__ void retrieveHashEntryIndexKernel(
		const unsigned* key_in, const unsigned key_size,
		const unsigned* table, const unsigned table_size,
		const uint2 primary_hash, const uint2 step_hash,
		unsigned* index_out
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < key_size) {
			const auto key = key_in[idx];
			const auto index = retrieveEntryIndexProbeHash(key, table, table_size, primary_hash, step_hash);
			index_out[idx] = index;
		}
	}

}; /* End of namespace device */
}; /* End of namespace hashing */


void hashing::CompactProbeTable::AllocateBuffer(const unsigned max_unique_key_size)
{
	const auto factor = 2;
	m_table_size = factor * max_unique_key_size;
	cudaSafeCall(cudaMalloc((void**)(&m_table), m_table_size * sizeof(unsigned)));
	cudaSafeCall(cudaMalloc((void**)(&m_failure_dev), sizeof(unsigned)));
	cudaSafeCall(cudaMallocHost((void**)(&m_failure_host), sizeof(unsigned)));
}


void hashing::CompactProbeTable::ResetTable(cudaStream_t stream)
{
	cudaSafeCall(cudaMemsetAsync(m_table, 0xff, m_table_size * sizeof(unsigned), stream));
	cudaSafeCall(cudaMemsetAsync(m_failure_dev, 0, sizeof(unsigned), stream));
	build_hash_constants(m_primary_hash, m_step_hash);
}

void hashing::CompactProbeTable::ReleaseBuffer()
{
	m_table_size = 0;
	cudaSafeCall(cudaFree(m_table));
	cudaSafeCall(cudaFree(m_failure_dev));
	cudaSafeCall(cudaFreeHost(m_failure_host));
}

bool hashing::CompactProbeTable::Insert(
	const unsigned * d_keys, const unsigned key_size,
	cudaStream_t stream
) {
	dim3 blk(64);
	dim3 grid(divUp(key_size, blk.x));
	device::compactProbeInsertEntryKernel<<<grid, blk, 0, stream>>>(
		d_keys, key_size, 
		m_table, m_table_size,
		m_primary_hash, m_step_hash, 
		m_failure_dev
	);

	//Sync before read the memory
	cudaSafeCall(cudaMemcpyAsync(m_failure_host, m_failure_dev, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
	cudaSafeCall(cudaStreamSynchronize(stream));
	return ((*m_failure_host) == 0);
}


void hashing::CompactProbeTable::RetrieveIndex(
	const unsigned * d_keys, const unsigned key_size,
	unsigned * d_index, 
	cudaStream_t stream
) {
	dim3 blk(64);
	dim3 grid(divUp(key_size, blk.x));
	device::retrieveHashEntryIndexKernel<<<grid, blk, 0, stream>>>(
		d_keys, key_size, 
		m_table, m_table_size, 
		m_primary_hash, m_step_hash,
		d_index
	);
}
