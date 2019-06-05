#include "common/common_types.h"
#include "common/sanity_check.h"
#include "common/algorithm_types.cuh"

#include <assert.h>
#include <device_launch_parameters.h>

namespace surfelwarp { namespace device {
    void checkCudaIndexTexture(
            cudaTextureObject_t vertex,
            cudaTextureObject_t normal,
            cudaTextureObject_t index,
            const DeviceArray<float4> &vertex_array,
            const DeviceArray<float4> &normal_array
    ) {

    }

} /*End of device*/ }; /*End of surfelwarp*/



void surfelwarp::checkPrefixSum() {
    //The size of testing
	std::vector<unsigned> test_sizes;
	test_sizes.push_back(1000000); test_sizes.push_back(3000000);

	PrefixSum prefix_sum;

	for (int j = 0; j < test_sizes.size(); j++) {
		//Construct the tests
		int test_size = test_sizes[j];
		std::vector<unsigned> in_array_host;
		in_array_host.resize(test_size);
		for (auto i = 0; i < in_array_host.size(); i++) {
			in_array_host[i] = rand() % 100;
		}

		//Upload it to device
		DeviceArray<unsigned> in_array;
		in_array.upload(in_array_host);

		//Do inclusive prefixsum on device
		prefix_sum.InclusiveSum(in_array);

		//Check the result
		std::vector<unsigned> prefixsum_array_host;
		prefix_sum.valid_prefixsum_array.download(prefixsum_array_host);
		assert(prefixsum_array_host.size() == in_array_host.size());
		int sum = 0;
		for (auto i = 0; i < in_array_host.size(); i++) {
			sum += in_array_host[i];
			assert(sum == prefixsum_array_host[i]);
		}

		//Do exclusive sum on device and check
		prefix_sum.ExclusiveSum(in_array);
		prefix_sum.valid_prefixsum_array.download(prefixsum_array_host);
		sum = 0;
		for (auto i = 0; i < in_array_host.size(); i++) {
			assert(sum == prefixsum_array_host[i]);
			sum += in_array_host[i];
		}
	}
}

void surfelwarp::checkKeyValueSort() {
	//The vector of test size
	std::vector<int> test_sizes;
	test_sizes.push_back(1000000); test_sizes.push_back(3000000);

	//Construct the sorter
	KeyValueSort<int, int> kv_sorter;

	//Do testing
	for (auto j = 0; j < test_sizes.size(); j++) {
		int test_size = test_sizes[j];

		//Construct the inputs at host
		std::vector<int> key_in_host, value_in_host;
		key_in_host.resize(test_size);
		value_in_host.resize(test_size);
		for (auto i = 0; i < key_in_host.size(); i++) {
			key_in_host[i] = rand() % test_size;
			value_in_host[i] = key_in_host[i] + 10;
		}

		//Upload them to device
		DeviceArray<int> key_in;
		key_in.upload(key_in_host);
		DeviceArray<int> value_in;
		value_in.upload(value_in_host);

		//Sort it
		kv_sorter.Sort(key_in, value_in);

		//Download the result
		std::vector<int> key_sorted_host, value_sorted_host;
		kv_sorter.valid_sorted_key.download(key_sorted_host);
		kv_sorter.valid_sorted_value.download(value_sorted_host);

		//Check the result
		for (auto i = 0; i < key_sorted_host.size() - 1; i++) {
			assert(key_sorted_host[i] <= key_sorted_host[i + 1]);
			assert(value_sorted_host[i] == key_sorted_host[i] + 10);
		}
	}
}

void surfelwarp::checkFlagSelection() {
    std::vector<int> test_sizes;
	test_sizes.push_back(1000000); test_sizes.push_back(3000000);
    //The selector
    FlagSelection flag_selector;

	for(auto i = 0; i < test_sizes.size(); i++){
        //Construct the test input
		int test_size = test_sizes[i];
        int num_selected = 0;
        std::vector<char> flag_host; flag_host.resize(test_size);
        for(auto j = 0; j < flag_host.size(); j++){
            flag_host[j] = (char)(rand() % 2);
            assert(flag_host[j] >= 0);
            assert(flag_host[j] <= 1);
            if(flag_host[j] == 1) num_selected++;
        }

        //Perform selection
        DeviceArray<char> flags; flags.upload(flag_host);
        flag_selector.Select(flags);

        //Check the result
        std::vector<int> selected_idx_host;
        flag_selector.valid_selected_idx.download(selected_idx_host);
        assert(selected_idx_host.size() == num_selected);
        for(auto j = 0;j < selected_idx_host.size(); j++){
            assert(flag_host[selected_idx_host[j]] == 1);
        }
	}
}

void surfelwarp::checkUniqueSelection() {
    std::vector<int> test_sizes;
    test_sizes.push_back(1000000); test_sizes.push_back(3000000);
    //The selector
    UniqueSelection unique_selector;

    for(auto i = 0; i < test_sizes.size(); i++){
        //Construct the test input
        int test_size = test_sizes[i];
        int num_selected = 0;
        std::vector<int> key_host; key_host.resize(test_size);
        for(auto j = 0; j < key_host.size(); j++){
            key_host[j] = (int)(rand() % 200);
        }

        //Perform selection
        DeviceArray<int> d_keys_in; d_keys_in.upload(key_host);
        unique_selector.Select(d_keys_in);

        //Check the result: the size shall be almost 200
        num_selected = unique_selector.valid_selected_element.size();
        assert(num_selected >= 198);
    }
}











