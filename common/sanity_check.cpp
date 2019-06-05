#include "common/sanity_check.h"
#include "common/common_utils.h"
#include "common/encode_utils.h"
#include <random>
#include <time.h>

void surfelwarp::fillRandomVector(std::vector<float> &vec) {
    std::default_random_engine generator((unsigned int) time(NULL));
    std::uniform_real_distribution<float> distribution;
    for (auto i = 0; i < vec.size(); i++) {
        vec[i] = distribution(generator);
    }
}

void surfelwarp::fillRandomVector(std::vector<unsigned int> &vec) {
	std::default_random_engine generator((unsigned int) time(NULL));
	std::uniform_int_distribution<unsigned int> distribution;
	for (auto i = 0; i < vec.size(); i++) {
		vec[i] = distribution(generator);
	}
}

void surfelwarp::test_encoding(const size_t test_size) {
	std::default_random_engine generator((unsigned int) time(NULL));
	std::uniform_int_distribution<unsigned short> distribution;
	for(auto i = 0; i < test_size; i++) {
		//First encode it
		unsigned char r, g, b, a;
		r = distribution(generator);
		g = distribution(generator);
		b = distribution(generator);
		a = distribution(generator);
		float encoded = float_encode_rgba(r, g, b, a);
        //Next decode it
        unsigned char decode_r, decode_g,decode_b,decode_a;
        float_decode_rgba(encoded, decode_r, decode_g, decode_b, decode_a);
        assert(r == decode_r);
        assert(g == decode_g);
        assert(b == decode_b);
        assert(a == decode_a);
    }
}

void surfelwarp::fillZeroVector(std::vector<float> &vec) {
    for (auto i = 0; i < vec.size(); i++) {
        vec[i] = 0.0f;
    }
}

double surfelwarp::maxRelativeError(
	const std::vector<float>& vec_0,
	const std::vector<float>& vec_1,
	const float small_cutoff,
	bool log_output
) {
	auto max_relaive_err = 0.0f;
	for (auto j = 0; j < std::min(vec_0.size(), vec_1.size()); j++) {
		float value_0 = vec_0[j];
		float value_1 = vec_1[j];
		float err = std::abs(value_0 - value_1);
		if (err > small_cutoff) {
			float relative_err = std::abs(err / std::max(std::abs(value_0), std::abs(value_1)));
			if (relative_err > max_relaive_err) {
				max_relaive_err = relative_err;
			}
			if(relative_err > 1e-3) {
				if(log_output) LOG(INFO) << "The relative error for " << j << " element is " << relative_err << " between " << vec_0[j] << " and " << vec_1[j];
			}
		}
		
	}
	return max_relaive_err;
}

double surfelwarp::maxRelativeError(const DeviceArray<float>& vec_0, const DeviceArray<float>& vec_1, const float small_cutoff)
{
	std::vector<float> h_vec_0, h_vec_1;
	vec_0.download(h_vec_0);
	vec_1.download(h_vec_1);
	return maxRelativeError(h_vec_0, h_vec_1, small_cutoff);
}

void surfelwarp::randomShuffle(std::vector<unsigned int> &key, std::vector<unsigned int> &value) {
    std::srand(time(nullptr));
    for (auto i = key.size() - 1; i > 0; --i) {
        const auto swap_idx = std::rand() % (i + 1);
        std::swap(key[i], key[swap_idx]);
        std::swap(value[i], value[swap_idx]);
        // rand() % (i+1) isn't actually correct, because the generated number
        // is not uniformly distributed for most values of i. A correct implementation
        // will need to essentially reimplement C++11 std::uniform_int_distribution,
        // which is beyond the scope of this example.
    }
}

void surfelwarp::randomShuffle(std::vector<unsigned>& vec)
{
	std::srand(time(nullptr));
	for (auto i = vec.size() - 1; i > 0; --i) {
		const auto swap_idx = std::rand() % (i + 1);
		std::swap(vec[i], vec[swap_idx]);
		// rand() % (i+1) isn't actually correct, because the generated number
		// is not uniformly distributed for most values of i. A correct implementation
		// will need to essentially reimplement C++11 std::uniform_int_distribution,
		// which is beyond the scope of this example.
	}
}


void surfelwarp::fillMultiKeyValuePairs(
	std::vector<unsigned int> &h_keys,
	const unsigned int num_entries,
	const unsigned int key_maximum,
	const unsigned int average_duplication
) {
	//The random number generator
	std::default_random_engine generator((unsigned int) time(NULL));
	std::uniform_int_distribution<int> distribution;
	
	//Insert into the key array
	h_keys.clear();
	int remains_entries = num_entries;
	while (remains_entries > 0) {
		auto key = distribution(generator) % key_maximum;
		auto key_duplication = distribution(generator) % (2 * average_duplication);
		if(key_duplication > remains_entries) key_duplication = remains_entries;
		remains_entries -= key_duplication;
		for(auto i = 0; i < key_duplication; i++) {
			h_keys.push_back(key);
		}
	}
	
	//Perform a random shuffle
	std::random_shuffle(h_keys.begin(), h_keys.end());
}


void surfelwarp::fillRandomLatticeKey(std::vector<LatticeCoordKey<5>>& lattice, const unsigned num_unique_elements, const unsigned duplicate)
{
	//The random number generator
	std::default_random_engine generator((unsigned int) time(NULL));
	std::uniform_int_distribution<int> distribution;

	//Insert into the lattice array
	lattice.clear();
	for(auto i = 0; i < num_unique_elements; i++) {
		LatticeCoordKey<5> coord;
		for(auto j = 0;j < 5; j++ ) {
			coord.key[j] = short(distribution(generator) % 1000);
		}
		for(auto j = 0; j < duplicate; j++) {
			lattice.push_back(coord);
		}
	}

	//Perform a random shuffle
	std::srand(time(nullptr));
	for (auto i = lattice.size() - 1; i > 0; --i) {
		const auto swap_idx = std::rand() % (i + 1);
		std::swap(lattice[i], lattice[swap_idx]);
	}
}

bool surfelwarp::isElementUnique(const std::vector<unsigned>& vec, const unsigned empty)
{
	std::vector<unsigned> sorted_vec(vec);
	std::sort(sorted_vec.begin(), sorted_vec.end());
	for(auto i = 0; i < sorted_vec.size() - 1; i++) {
		if(sorted_vec[i] == sorted_vec[i + 1] && sorted_vec[i] != empty) {
			return false;
		}
	}
	return true;
}

unsigned surfelwarp::numUniqueElement(const std::vector<unsigned>& vec, const unsigned empty)
{
	//First do a sorting
	std::vector<unsigned> sorted_vec(vec);
	std::sort(sorted_vec.begin(), sorted_vec.end());

	//Count the elements
	unsigned unique_elements = 1; //The first element is assumed to be unique
	for (auto i = 1; i < sorted_vec.size(); i++) {
		if (sorted_vec[i] == sorted_vec[i - 1] || sorted_vec[i] == empty) {
			//duplicate or invalid elements
		} else {
			unique_elements++;
		}
	}

	return unique_elements;
}

unsigned surfelwarp::numUniqueElement(const surfelwarp::DeviceArray<unsigned int> &vec, const unsigned empty) {
	std::vector<unsigned> h_vec;
	vec.download(h_vec);
	return numUniqueElement(h_vec, empty);
}

unsigned surfelwarp::numNonZeroElement(const std::vector<unsigned> &vec) {
	unsigned nonzero_count = 0;
	for(auto i = 0; i < vec.size(); i++) {
		if (vec[i] != 0) {
			nonzero_count++;
		}
	}
	return nonzero_count;
}

unsigned surfelwarp::numNonZeroElement(const DeviceArray<unsigned int> &vec) {
	std::vector<unsigned> h_vec;
	vec.download(h_vec);
	return numNonZeroElement(h_vec);
}

unsigned surfelwarp::numNonZeroElement(const surfelwarp::DeviceArrayView<unsigned int> &vec) {
	std::vector<unsigned> h_vec;
	vec.Download(h_vec);
	return numNonZeroElement(h_vec);
}

bool surfelwarp::containsNaN(const std::vector<float4> &vec) {
	for(auto i = 0; i < vec.size(); i++) {
		const float4 element = vec[i];
		if(std::isnan(element.x)) return true;
		if(std::isnan(element.y)) return true;
		if(std::isnan(element.z)) return true;
		if(std::isnan(element.w)) return true;
	}
	return false;
}

bool surfelwarp::containsNaN(const surfelwarp::DeviceArrayView<float4> &vec) {
	std::vector<float4> h_vec;
	vec.Download(h_vec);
	return containsNaN(h_vec);
}

void surfelwarp::applyRandomSE3ToWarpField(
	std::vector<DualQuaternion> &node_se3,
	float max_rot, float max_trans
) {
	std::default_random_engine generator((unsigned int) time(NULL));
	std::uniform_real_distribution<float> rot_distribution(-max_rot, max_rot);
	std::uniform_real_distribution<float> trans_distribution(-max_trans, max_trans);
	for(auto i = 0; i < node_se3.size(); i++) {
		float3 twist_rot, twist_trans;
		twist_rot.x = rot_distribution(generator);
		twist_rot.y = rot_distribution(generator);
		twist_rot.z = rot_distribution(generator);
		twist_trans.x = trans_distribution(generator);
		twist_trans.y = trans_distribution(generator);
		twist_trans.z = trans_distribution(generator);
		apply_twist(twist_rot, twist_trans, node_se3[i]);
	}
}

void surfelwarp::applyRandomSE3ToWarpField(
	DeviceArraySlice<DualQuaternion> node_se3,
	float max_rot, float max_trans
) {
	std::vector<DualQuaternion> h_node_se3;
	node_se3.SyncToHost(h_node_se3);
	applyRandomSE3ToWarpField(h_node_se3, max_rot, max_trans);
	node_se3.SyncFromHost(h_node_se3);
}


double surfelwarp::differenceL2(const std::vector<float> &vec_0, const std::vector<float> &vec_1) {
	double diff_square = 0.0;
	for(auto i = 0; i < std::min(vec_0.size(), vec_1.size()); i++) {
		const auto diff = vec_0[i] - vec_1[i];
		diff_square += (diff * diff);
	}
	return std::sqrt(diff_square);
}


double surfelwarp::differenceL2(const DeviceArray<float> &vec_0, const DeviceArray<float> &vec_1) {
	std::vector<float> h_vec_0, h_vec_1;
	vec_0.download(h_vec_0);
	vec_1.download(h_vec_1);
	return differenceL2(h_vec_0, h_vec_1);
}


//The statistic method about any residual vector, the value might be negative
void surfelwarp::residualVectorStatistic(const std::vector<float> &residual_in, int topk, std::ostream& output) {
	//Apply abs to all input
	std::vector<float> sorted_residual_vec;
	sorted_residual_vec.clear();
	sorted_residual_vec.resize(residual_in.size());

	double average_residual = 0.0;
	for(auto i = 0; i < residual_in.size(); i++) {
		sorted_residual_vec[i] = std::abs(residual_in[i]);
		average_residual += sorted_residual_vec[i];
	}
	
	output << "The average of residual is " << average_residual / residual_in.size() << std::endl;
	
	//Sort it
	std::sort(sorted_residual_vec.begin(), sorted_residual_vec.end());
	
	//Max, min and medium
	const auto mid_idx = residual_in.size() >> 1;
	output << "The max, middle and min of residual is "
	       << sorted_residual_vec[sorted_residual_vec.size() - 1]
	       << " " << sorted_residual_vec[mid_idx]
	       << " " << sorted_residual_vec[0] << std::endl;
	
	//The top k residual
	output << "The top " << topk << " residual is";
	for(auto i = 0; i < topk; i++) {
		auto idx = sorted_residual_vec.size() - 1 - i;
		if(idx >= 0 && idx < sorted_residual_vec.size()) {
			output << " " << sorted_residual_vec[idx];
		}
	}
	output << std::endl;
}

