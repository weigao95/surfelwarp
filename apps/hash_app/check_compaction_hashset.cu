#include "check_compaction_hashset.h"
#include "common/sanity_check.h"
#include "hashing/TicketBoardSet.cuh"
#include "imgproc/segmentation/permutohedral_common.h"

#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
	
	__global__ void checkCompactionHashSetInsertKernel(
		const unsigned* d_keys, const unsigned key_size,
		unsigned* table, const unsigned table_size,
		const uint2 primary_hash, const uint2 step_hash,
		unsigned* failure_flag
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < key_size) {
			const unsigned key = d_keys[idx];
			const bool success = hashing::device::insertHashSetUnsignedEntry(
				key, 
				table, table_size, 
				primary_hash, step_hash
			);
			if(!success) {
				atomicAdd(failure_flag, 1);
			}
		}
	}

	__global__ void checkPermutohedralKerInsertKernel(
		const LatticeCoordKey<5>* d_lattice, const unsigned num_entries,
		unsigned* ticket_board, 
		LatticeCoordKey<5>* table, const unsigned table_size,
		const uint2 primary_hash, const uint2 step_hash,
		unsigned* flag
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < num_entries) {
			const auto lattice_coord = d_lattice[idx];
			const auto hashed_coord = lattice_coord.hash();
			const bool success = hashing::device::insertTicketSetEntry<LatticeCoordKey<5>>(
				lattice_coord, hashed_coord, 
				ticket_board, table, table_size, 
				primary_hash, step_hash, flag
			);
			if(!success) {
				atomicAdd(flag, 1);
			}
		}
	}
	
	__global__ void checkRetrieveCompactedIndexKernel(
		const LatticeCoordKey<5>* d_lattice, const unsigned num_entries,
		const unsigned* ticket_board, const unsigned* compacted_index,
		LatticeCoordKey<5>* table, const unsigned table_size,
		const uint2 primary_hash, const uint2 step_hash,
		unsigned* index_out
	){
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < num_entries) {
			const auto lattice_coord = d_lattice[idx];
			const auto hashed_coord = lattice_coord.hash();
			const auto index = hashing::device::retrieveTicketSetKeyIndex<LatticeCoordKey<5>>(
				lattice_coord, hashed_coord,
				ticket_board, table, table_size,
				primary_hash, step_hash
			);
			unsigned compacted = 0xffffffffu;
			if(index != 0xffffffffu) {
				compacted = compacted_index[index];
			}
			index_out[idx] = compacted;
		}
	}

	
	__global__ void checkBuildCompactedElementKernel(
		const unsigned* compacted_index, const LatticeCoordKey<5>* table, unsigned table_size,
		LatticeCoordKey<5>* compacted_lattice
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < table_size) {
			const auto offset = compacted_index[idx];
			if(offset != 0xffffffffu) {
				compacted_lattice[offset] = table[idx];
			}
		}
	}

}; /* End of namespace device */
}; /* End of namespace surfelwarp */

void surfelwarp::check_compaction_hashset(const size_t test_size, const size_t test_iters)
{
	for(auto i = 0; i < test_iters; i++) {
		check_compaction_hashset(test_size);
		if(i % 10 == 0) {
			std::cout << i << " th test!" << std::endl;
		}
	}
}

void surfelwarp::check_compaction_hashset(const size_t test_size)
{
	using namespace hashing;

	//Prepare the data
	std::vector<unsigned> h_keys;
	const unsigned duplicate = 300;
	const unsigned unique_keys = divUp(test_size, duplicate);
	fillMultiKeyValuePairs(h_keys, test_size, 2 * test_size, duplicate);
	DeviceArray<unsigned> d_keys;
	d_keys.upload(h_keys);

	//Prepare the table
	CompactionHashSet set;
	set.AllocateBuffer(unique_keys);
	set.ResetTable();

	//Prepare for failure flag
	DeviceArray<unsigned> d_failure;
	std::vector<unsigned> h_failure;
	h_failure.push_back(0);
	d_failure.upload(h_failure);

	//Do inseration
	dim3 blk(128);
	dim3 grid(divUp(test_size, blk.x));
	device::checkCompactionHashSetInsertKernel<<<grid, blk>>>(
		d_keys.ptr(), d_keys.size(), 
		set.table, set.table_size, 
		set.primary_hash, set.step_hash,
		d_failure.ptr()
	);

	//Download the failure flag
	d_failure.download(h_failure);
	assert(h_failure[0] == 0);

	//Check the uniqueness
	std::vector<unsigned> h_table_entry;
	h_table_entry.resize(set.table_size);
	cudaMemcpy(h_table_entry.data(), set.table, sizeof(unsigned) * set.table_size, cudaMemcpyDeviceToHost);
	assert(isElementUnique(h_table_entry, 0xffffffff));

	const auto num_unique_elements_in = numUniqueElement(h_keys, 0xffffffffu);
	//std::cout << "The num of output unique elements is " << num_unique_elements_in << std::endl;
	const auto num_unique_elements_out = numUniqueElement(h_table_entry, 0xffffffffu);
	assert(num_unique_elements_in == num_unique_elements_out);

	//Build the index
	set.BuildIndex();

	//Clear the set
	set.ReleaseBuffer();
}

void surfelwarp::check_permutohedral_set(const size_t test_size, const size_t test_iters)
{
	for(auto i = 0; i < test_iters; i++) {
		check_permutohedral_set(test_size);
		if(i % 10 == 0) {
			std::cout << i << " th test!" << std::endl;
		}
	}
}

void surfelwarp::check_permutohedral_set(const size_t test_size)
{
	using namespace hashing;

	//Prepare the data
	std::vector<LatticeCoordKey<5>> h_lattice;
	const unsigned duplicate = 64;
	const auto num_unique_elems = divUp(test_size, duplicate);
	fillRandomLatticeKey(h_lattice, num_unique_elems, duplicate);

	//Upload it 
	DeviceArray<LatticeCoordKey<5>> d_lattice;
	d_lattice.upload(h_lattice);

	//Prepare the table
	TicketBoardSet<LatticeCoordKey<5>> set;
	set.AllocateBuffer(unsigned(num_unique_elems * 1.5));
	
	//Empty the table
	LatticeCoordKey<5> empty; empty.set_null();
	set.ResetTable(empty);
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());

	//Prepare for failure flag
	DeviceArray<unsigned> d_failure;
	std::vector<unsigned> h_failure;
	h_failure.push_back(0);
	d_failure.upload(h_failure);

	//Do inseration
	dim3 blk(128);
	dim3 grid(divUp(test_size, blk.x));
	device::checkPermutohedralKerInsertKernel<<<grid, blk>>>(
		d_lattice.ptr(), test_size, 
		set.TicketBoard(), set.Table(), set.TableSize(), 
		set.PrimaryHash(), set.StepHash(), 
		d_failure.ptr()
	);
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());

	//Download the failure flag
	d_failure.download(h_failure);
	if(h_failure[0] != 0) {
		std::cout << "Insertation failed, key lost" << std::endl;
	}
	
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
	
	//Get the number of valid and unique tickets
	std::vector<unsigned> h_tickets;
	h_tickets.resize(set.TableSize());
	cudaMemcpy(h_tickets.data(), set.TicketBoard(), sizeof(unsigned) * set.TableSize(), cudaMemcpyDeviceToHost);
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
	
	//Build the index
	set.BuildIndex();

#if 0
	//Check the actual elements
	const bool check_elements = false;
	if(check_elements) {
		//Might need more check
		//std::cout << "Check the actual elements" << std::endl;
		
		//Check it
		std::vector<LatticeCoordKey<5>> h_lattice_keys;
		h_lattice_keys.resize(set.TableSize());
		cudaMemcpy(h_lattice_keys.data(), set.Table(), sizeof(LatticeCoordKey<5>) * set.TableSize(), cudaMemcpyDeviceToHost);
		cudaSafeCall(cudaDeviceSynchronize());
		cudaSafeCall(cudaGetLastError());
		std::cout << "The inserted duplicate is " << averageDuplicate(h_lattice_keys, empty) << std::endl;
	}
#endif
	
	//And check the uniqueness
#if 0
	std::vector<unsigned> h_unique, h_compacted_index, h_compact_index_from_unique;
	std::vector<LatticeCoordKey<5>> h_lattice_from_unique, h_table;
	h_lattice_from_unique.clear();
	
	h_compacted_index.resize(set.TableSize());
	h_compact_index_from_unique.resize(set.TableSize());
	h_unique.resize(set.TableSize());
	h_table.resize(set.TableSize());
	cudaMemcpy(h_unique.data(), set.UniqueIndicator(), sizeof(unsigned) * set.TableSize(), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_table.data(), set.Table(), sizeof(LatticeCoordKey<5>) * set.TableSize(), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_compacted_index.data(), set.CompactedIndex(), sizeof(unsigned) * set.TableSize(), cudaMemcpyDeviceToHost);
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());
	int unique_count = 0;
	for(auto i = 0; i < h_unique.size(); i++) {
		h_compact_index_from_unique[i] = 0xffffffffu;
		if(h_unique[i] > 0) {
			h_lattice_from_unique.push_back(h_table[i]);
			h_compact_index_from_unique[i] = unique_count;
			unique_count++;
		}
		
		//Check it
		assert(h_compacted_index[i] == h_compact_index_from_unique[i]);
	}
	assert(unique_count == num_unique_elems);
	assert(isElementUniqueNonEmptyNaive(h_lattice_from_unique, empty));
#endif
	
	//Do a compaction
	DeviceArray<LatticeCoordKey<5>> compacted_lattice;
	compacted_lattice.create(num_unique_elems);
	device::checkBuildCompactedElementKernel<<<grid, blk>>>(
		set.CompactedIndex(), set.Table(), set.TableSize(), 
		compacted_lattice.ptr()
	);
	std::vector<LatticeCoordKey<5>> h_compacted_lattice;
	compacted_lattice.download(h_compacted_lattice);
	assert(isElementUniqueNonEmptyNaive(h_compacted_lattice, empty));
	
	//Check the compacted index
	DeviceArray<unsigned> compacted_index_out;
	compacted_index_out.create(d_lattice.size());

	device::checkRetrieveCompactedIndexKernel<<<grid, blk>>>(
		d_lattice.ptr(), d_lattice.size(), 
		set.TicketBoard(), set.CompactedIndex(), set.Table(), set.TableSize(), 
		set.PrimaryHash(), set.StepHash(), 
		compacted_index_out.ptr()
	);
	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaGetLastError());

	//Check the result
	std::vector<unsigned> h_compacted_index_out;
	compacted_index_out.download(h_compacted_index_out);
	for(auto i = 0; i < h_compacted_index_out.size(); i++) {
		const auto idx = h_compacted_index_out[i];
		assert(idx != 0xffffffffu);
		if(!(h_lattice[i] == h_compacted_lattice[idx])) {
			std::cout << "Incorrect at" << i << std::endl;
			assert(false);
		}
	}

	set.ReleaseBuffer();
}