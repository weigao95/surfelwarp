#include "sort_compaction.h"
#include "common/sanity_check.h"
#include <vector>
#include <device_launch_parameters.h>
#include <pcl/common/time.h>

namespace surfelwarp { namespace device {
	
	__global__ void markUniqueElementKernel(
		const PtrSz<const unsigned> sorted_key,
		PtrSz<unsigned> unique_indicator
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < sorted_key.size) {
			unsigned unique = 0;
			if(idx == 0 || sorted_key[idx] != sorted_key[idx - 1]) {
				unique = 1;
			}
			unique_indicator[idx] = unique;
		}
	}


	__global__ void compactElementKernel(
		const PtrSz<const unsigned> sorted_key_array,
		const PtrSz<const unsigned> unique_indicator,
		const PtrSz<const unsigned> prefixsum_indicator,
		PtrSz<unsigned> compacted_array
	) {
		const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
		if(idx < unique_indicator.size) {
			const auto offset = prefixsum_indicator[idx] - 1;
			compacted_array[offset] = sorted_key_array[idx];
		}
	}

}; /* End of namespace device */
}; /* End of namespace srufelwarp */

surfelwarp::DeviceArray<unsigned> surfelwarp::sortCompact(
	const DeviceArray<unsigned> &array_in,
	KeyValueSort<unsigned , unsigned>& sorter,
	DeviceArray<unsigned>& valid_indicator,
	PrefixSum &prefixsum,
	DeviceArray<unsigned>& compacted_buffer
) {
	//First sort it
	sorter.Sort(array_in);
	
	//Mark the changed value
	dim3 blk(128);
	dim3 grid(divUp(array_in.size(), blk.x));
	device::markUniqueElementKernel<<<grid, blk>>>(sorter.valid_sorted_key, valid_indicator);
	
	//Do a prefix sum
	prefixsum.InclusiveSum(valid_indicator);

	//Copy the size
	int h_compacted_size;
	cudaMemcpy(&h_compacted_size, prefixsum.valid_prefixsum_array.ptr() + prefixsum.valid_prefixsum_array.size() - 1, sizeof(unsigned), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	//Construct the size-correct compacted array
	DeviceArray<unsigned> valid_compacted_array = DeviceArray<unsigned>(compacted_buffer.ptr(), h_compacted_size);

	//Compact it
	device::compactElementKernel<<<grid, blk>>>(
		sorter.valid_sorted_key, 
		valid_indicator, 
		prefixsum.valid_prefixsum_array,
		valid_compacted_array
	);
	return valid_compacted_array;
}

void surfelwarp::allocateSortCompactBuffer(
	size_t input_size,
	KeyValueSort<unsigned int, unsigned int> &sort,
	PrefixSum &prefixSum
) {
	sort.AllocateBuffer(input_size);
	prefixSum.AllocateBuffer(input_size);
}



void surfelwarp::checkSortCompactionPerformance(
	const DeviceArray<unsigned>& array_in, 
	const size_t test_iters
) {
	//Prepare data
	KeyValueSort<unsigned, unsigned> sort;
	PrefixSum prefix_sum;
	DeviceArray<unsigned> unique_indicator, compacted_buffer;
	unique_indicator.create(array_in.size());
	compacted_buffer.create(array_in.size());
	allocateSortCompactBuffer(array_in.size(), sort, prefix_sum);

	//The performance testing loop
	{
		pcl::ScopeTime time("Perform compaction using sorting");
		for(auto i = 0; i < test_iters; i++) {
			auto compacted_array = sortCompact(
				array_in, 
				sort, 
				unique_indicator, 
				prefix_sum,
				compacted_buffer
			);

			//Check correctness once
			//std::vector<unsigned> h_compacted_keys;
			//compacted_array.download(h_compacted_keys);
			//assert(isElementUnique(h_compacted_keys, 0xffffffff));
		}
		cudaDeviceSynchronize();
	}
}