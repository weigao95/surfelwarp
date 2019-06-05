#include "common/device_intrinsics.h"
#include "imgproc/segmentation/foreground_permutohedral_deduplicate.h"
#include "imgproc/segmentation/crf_config.h"
#include "imgproc/segmentation/crf_common.h"
#include "imgproc/segmentation/permutohedral_common.h"

#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math_functions.h>

namespace surfelwarp { namespace device {
	
	
	enum {
		num_threads = 32
	};

	__global__ void foregroundDeduplicateHashedLatticeKernel(
		const unsigned* compacted_hash_offset,
		unsigned* sorted_lattice_index,
		const LatticeCoordKey<image_permutohedral_dim>* lattice_coord_key_array,
		ForegroundPermutohedralLatticePerHash* compacted_lattice_record,
		unsigned* global_duplicate_flag,
		unsigned* local_reduce_buffer
	) {
		//Do not need check the range
		const auto hashed_key_index = blockIdx.x;

		//The begin and end in sorted_lattice_index
		const auto begin = compacted_hash_offset[hashed_key_index + 0];
		const auto end = compacted_hash_offset[hashed_key_index + 1];

		//The shared lattice coord is init by the first threads
		__shared__ LatticeCoordKey<image_permutohedral_dim> unique_lattice[max_lattice_per_hash];
		__shared__ short num_unique_lattice_shared;
		if (threadIdx.x == 0) {
			for (auto i = 0; i < max_lattice_per_hash; i++) {
				unique_lattice[i].set_null();
			}
		}

		//Sync here
		__syncthreads();

		//Shared lattice for cooperative loading
		__shared__ LatticeCoordKey<image_permutohedral_dim> lattice[num_threads];

		//Only thread 0 hold a valid value
		short num_unique_lattice = 0;

		//First loop, load all possible lattice in this range, check if unique
		for (auto i = begin; i < end; i += num_threads) {
			//Cooperative loading
			const auto thread_i = i + threadIdx.x;
			if (thread_i < end) {
				const auto index = sorted_lattice_index[thread_i];
				lattice[threadIdx.x] = lattice_coord_key_array[index];
			}
			else {
				lattice[threadIdx.x].set_null();
			}

			//Sync here for cooperative loading
			__syncthreads();

			//Let one thread check the uniqueness
			if (threadIdx.x == 0) {
				for (auto j = 0; j < num_threads; j++) {
					const auto lattice_for_check = lattice[j];
					//bool null_lattice = lattice_for_check.is_null();
					bool new_lattice = !(lattice_for_check.is_null());

					//Checking loop
					for (auto k = 0; k < min(num_unique_lattice, max_lattice_per_hash); k++) {
						const auto existing_lattice = unique_lattice[k];
						if (existing_lattice.less_than(lattice_for_check) == 0) {
							new_lattice = false;
							break;
						}
					}

					//Update it
					if (new_lattice) {
						unique_lattice[num_unique_lattice] = lattice_for_check;
						num_unique_lattice++;
					}
				} //End of checking loop
			}
		}


		//End of checking loop
		if (threadIdx.x == 0) {
			num_unique_lattice_shared = num_unique_lattice;
		}
		__syncthreads();

		//Store the result and return
		if(num_unique_lattice_shared == 1) { 
			//Construct the result
			ForegroundPermutohedralLatticePerHash lattice_record;
			lattice_record.num_lattice = 1;
			lattice_record.lattice_coord_key[0] = unique_lattice[0];
			lattice_record.lattice_coord_key[1].set_null();
			lattice_record.lattice_coord_offset[0].x = begin;
			lattice_record.lattice_coord_offset[0].y = end;
			if(threadIdx.x == 0) {
				compacted_lattice_record[hashed_key_index] = lattice_record;
			}
			return; //All threads will return
		}

		//At least 2 elements, set the flag
		*global_duplicate_flag = 1;
		
		__shared__ int index_offset;
		if(threadIdx.x == 0) {
			index_offset = 0;
		}
		
		//Construct the result
		ForegroundPermutohedralLatticePerHash lattice_record;
		lattice_record.num_lattice = num_unique_lattice_shared;
		
		//The main deduplicate loop
		for(auto lattice_idx = 0; lattice_idx < num_unique_lattice_shared; lattice_idx++) {
			//The lattice for this index
			const LatticeCoordKey<image_permutohedral_dim> curr_unique_lattice = unique_lattice[lattice_idx];
			
			//Store input record
			lattice_record.lattice_coord_key[lattice_idx] = curr_unique_lattice;
			lattice_record.lattice_coord_offset[lattice_idx].x = begin + index_offset;
			
			//The main processing loop
			for(auto i = begin; i < end; i += num_threads) {
				//Cooperative loading
				const auto thread_i = i + threadIdx.x;
				LatticeCoordKey<image_permutohedral_dim> lattice_thread;
				int lattice_matched = 0;
				if (thread_i < end) {
					const auto index = sorted_lattice_index[thread_i];
					lattice_thread = lattice_coord_key_array[index];
					lattice_matched = (curr_unique_lattice.less_than(lattice_thread) == 0);
				}
				
				//Do a warp scan on matched
				int scanned_matched = lattice_matched;
				scanned_matched = warp_scan(scanned_matched);
				
				//Store it
				if(lattice_matched) {
					const auto thread_offset = begin + index_offset + scanned_matched - 1;
					local_reduce_buffer[thread_offset] = sorted_lattice_index[thread_i];
				}
				
				//Increase on the global offset
				if(threadIdx.x == 31) {
					index_offset += scanned_matched;
				}
				__syncthreads();
			}
			
			//Store the result
			lattice_record.lattice_coord_offset[lattice_idx].y = begin + index_offset;
		}

		//Copy the local reduce buffer back to sorted buffer
		for(auto i = begin; i < end; i += num_threads) {
			//Cooperative loading
			const auto thread_i = i + threadIdx.x;
			if(thread_i < end) {
				sorted_lattice_index[thread_i] = local_reduce_buffer[thread_i];
			}
		}
	}

}; /* End of namespace device */
}; /* End of namespace surfelwarp */


void surfelwarp::foregroundDeduplicateHashedLattice(
	const DeviceArray<unsigned>& compacted_hash_offset, 
	DeviceArray<unsigned>& sorted_lattice_index, 
	const DeviceArray<LatticeCoordKey<image_permutohedral_dim>>& lattice_coord_key_array,
	DeviceArray<ForegroundPermutohedralLatticePerHash>& compacted_lattice_record,
	unsigned* d_duplicate_flag,
	DeviceArray<unsigned>& deduplicate_reduce_buffer,
	cudaStream_t stream
) {
	dim3 blk(32);
	dim3 grid(compacted_hash_offset.size() - 1);
	device::foregroundDeduplicateHashedLatticeKernel<<<grid, blk, 0, stream>>>(
		compacted_hash_offset, 
		sorted_lattice_index, 
		lattice_coord_key_array, 
		compacted_lattice_record, 
		d_duplicate_flag,
		deduplicate_reduce_buffer
	);

	//Sync and check error for sorting
#if defined(CUDA_DEBUG_SYNC_CHECK)
	cudaSafeCall(cudaStreamSynchronize(stream));
	cudaSafeCall(cudaGetLastError());
#endif
}