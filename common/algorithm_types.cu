#include <cub/cub.cuh>
#include <device_launch_parameters.h>
#include "common/algorithm_types.h"

__host__ void surfelwarp::PrefixSum::AllocateBuffer(size_t input_size) {
    //Do not need allocation
    if (m_prefixsum_buffer.size() >= input_size) return;

	//If existing buffer. clear them
	if (m_prefixsum_buffer.size() > 0) {
		m_prefixsum_buffer.release();
		m_temp_storage.release();
	}

    //Do allocation
    m_prefixsum_buffer.create(input_size);
    //Query the temp storage for input size
    size_t prefixsum_bytes = 0;
    cub::DeviceScan::InclusiveSum(NULL, prefixsum_bytes,
                                  m_prefixsum_buffer.ptr(), m_prefixsum_buffer.ptr(), (int) input_size, 0);
    m_temp_storage.create(prefixsum_bytes);
    return;
}

void surfelwarp::PrefixSum::InclusiveSum(const DeviceArray<unsigned> &array_in, cudaStream_t stream, bool debug_sync) {
    //Allocate the buffer if not enough
    AllocateBuffer(array_in.size());

    //Construct the result array
    valid_prefixsum_array = DeviceArray<unsigned>(m_prefixsum_buffer.ptr(), array_in.size());

    //Do prefixsum
    size_t inclusive_sum_bytes = m_temp_storage.sizeBytes();
    cub::DeviceScan::InclusiveSum(m_temp_storage, inclusive_sum_bytes,
                                  array_in.ptr(), valid_prefixsum_array.ptr(), (int) array_in.size(), stream,
                                  debug_sync);
    return;
}

void
surfelwarp::PrefixSum::InclusiveSum(const surfelwarp::DeviceArrayView<unsigned int> &array_in, cudaStream_t stream) {
	//Allocate the buffer if not enough
	AllocateBuffer(array_in.Size());
	
	//Construct the result array
	valid_prefixsum_array = DeviceArray<unsigned>(m_prefixsum_buffer.ptr(), array_in.Size());
	
	//Do prefixsum
	size_t inclusive_sum_bytes = m_temp_storage.sizeBytes();
	cub::DeviceScan::InclusiveSum(
		m_temp_storage, inclusive_sum_bytes,
		array_in.RawPtr(), valid_prefixsum_array.ptr(),
		(int) array_in.Size(),
		stream, false
	);
	return;
}

void surfelwarp::PrefixSum::ExclusiveSum(const DeviceArray<unsigned> &array_in, cudaStream_t stream, bool debug_sync) {
    //Allocate the buffer if not enough
    AllocateBuffer(array_in.size());

    //Construct the result array
    valid_prefixsum_array = DeviceArray<unsigned>(m_prefixsum_buffer.ptr(), array_in.size());

    //Do prefixsum
    size_t exclusive_sum_bytes = m_temp_storage.sizeBytes();
    cub::DeviceScan::ExclusiveSum(m_temp_storage, exclusive_sum_bytes,
                                  array_in.ptr(), valid_prefixsum_array.ptr(), (int) array_in.size(), stream,
                                  debug_sync);
    return;
}




namespace surfelwarp { namespace device {

	/* The kernel to init the seleted_input, 
	   Make the selected output to be the index in the original array
	*/
    __global__ void selectionIndexInitKernel(
		PtrSz<int> selection_input_buffer
	) {
		const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
		if(idx < selection_input_buffer.size)
			selection_input_buffer.data[idx] = idx;
    }

} /*End of namespace device*/ } /*End of namespace surfelwarp*/



void surfelwarp::FlagSelection::AllocateAndInit(size_t input_size, cudaStream_t stream) {
	if (m_selection_input_buffer.size() >= input_size) return;
	
	//Release the memory if required
	if (m_selected_idx_buffer.size() > 0) {
		m_selected_idx_buffer.release();
		m_selection_input_buffer.release();
		m_temp_storage.release();
		select_indicator_buffer.release();
	}

	//Allocate new storages
	size_t allocate_size = 3 * input_size / 2;

	//Do allocation: the max size of selected index is the same as selection input
	m_selected_idx_buffer.create(allocate_size);
	m_selection_input_buffer.create(allocate_size);
	select_indicator_buffer.create(allocate_size);

	//Check the required temp storage
	size_t temp_storage_bytes = 0;
	cub::DeviceSelect::Flagged(
		NULL, temp_storage_bytes,
		m_selection_input_buffer.ptr(), m_selected_idx_buffer.ptr(),
		valid_selected_idx.ptr(), m_device_num_selected,
		allocate_size,
		stream
	);

	m_temp_storage.create(temp_storage_bytes);

	//Init the buffer
	dim3 blk(128);
	dim3 grid(divUp(m_selection_input_buffer.size(), blk.x));
	device::selectionIndexInitKernel<<<grid, blk, 0, stream>>>(m_selection_input_buffer);
}


void surfelwarp::FlagSelection::Select(
        const DeviceArray<char> &flags,
        cudaStream_t stream, bool debug_sync
) {
    //Do allocation and init
    AllocateAndInit(flags.size(), stream);

    //Construct the array for selection
    DeviceArray<int> selection_idx_input = DeviceArray<int>(m_selection_input_buffer.ptr(), flags.size());
    valid_selected_idx = DeviceArray<int>(m_selected_idx_buffer.ptr(), flags.size());

    //Do selection
    size_t temp_storage_bytes = m_temp_storage.sizeBytes();
    cub::DeviceSelect::Flagged(
            m_temp_storage.ptr(), temp_storage_bytes,
            selection_idx_input.ptr(), flags.ptr(),
            valid_selected_idx.ptr(), m_device_num_selected,
            (int)flags.size(),
            stream, debug_sync
    );

    //Need sync before host accessing
    cudaSafeCall(cudaMemcpyAsync(m_host_num_selected, m_device_num_selected, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaSafeCall(cudaStreamSynchronize(stream));

    //Correct the size of output
    valid_selected_idx = DeviceArray<int>(m_selected_idx_buffer.ptr(), *m_host_num_selected);
}


void surfelwarp::FlagSelection::SelectUnsigned(
	const DeviceArray<char> &flags, const DeviceArray<unsigned> &select_from,
	DeviceArray<unsigned> &select_to_buffer,
	DeviceArray<unsigned> &valid_selected_array,
	cudaStream_t stream
) {
	//Do allocation and init
	AllocateAndInit(flags.size(), stream);
	
	//Do selection
	size_t temp_storage_bytes = m_temp_storage.sizeBytes();
	cub::DeviceSelect::Flagged(
		m_temp_storage.ptr(), temp_storage_bytes,
		select_from.ptr(), flags.ptr(),
		select_to_buffer.ptr(), m_device_num_selected,
		(int)flags.size(),
		stream, false
	);
	
	//Need sync before host accessing
	cudaSafeCall(cudaMemcpyAsync(m_host_num_selected, m_device_num_selected, sizeof(int), cudaMemcpyDeviceToHost, stream));
	cudaSafeCall(cudaStreamSynchronize(stream));
	
	//Now it is safe to access the memcpy output
	valid_selected_array = DeviceArray<unsigned>(select_to_buffer.ptr(), *m_host_num_selected);
}


void surfelwarp::UniqueSelection::Allocate(size_t input_size) {
    if (m_selected_element_buffer.size() >= input_size) return;

    //Clear existing cache
    if(m_selected_element_buffer.size() > 0) {
        m_selected_element_buffer.release();
        m_temp_storage.release();
    }

    //Allocate new storages
    size_t allocate_size = 3 * input_size / 2;
    m_selected_element_buffer.create(allocate_size);

    //Query the required buffer
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Unique(
            m_temp_storage.ptr(), temp_storage_bytes,
            m_selected_element_buffer.ptr(), //The input and output are not used in querying
            m_selected_element_buffer.ptr(), m_device_num_selected,
            (int)m_selected_element_buffer.size()
    );

    m_temp_storage.create(temp_storage_bytes);
}

void surfelwarp::UniqueSelection::Select(const DeviceArray<int> &key_in, cudaStream_t stream, bool debug_sync) {
    //Check and allocate required buffer
    Allocate(key_in.size());


    //Do selection
    size_t temp_storage_bytes = m_temp_storage.sizeBytes();
    cub::DeviceSelect::Unique(
            m_temp_storage.ptr(), temp_storage_bytes,
            key_in.ptr(), //The input
            m_selected_element_buffer.ptr(), m_device_num_selected, //The output
            (int)key_in.size(), stream, debug_sync
    );

    //Need sync before host accessing
    cudaSafeCall(cudaMemcpyAsync(m_host_num_selected, m_device_num_selected, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaSafeCall(cudaStreamSynchronize(stream));

    //Construct the size-correct result
    valid_selected_element = DeviceArray<int>(m_selected_element_buffer.ptr(), *m_host_num_selected);
}