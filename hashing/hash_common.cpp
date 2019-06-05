#include "hashing/hash_common.h"
#include <ctime>
#include <random>
#include <algorithm>

void hashing::build_hash_constants(HashConstants & hash_constants, uint2 & stash_constants)
{
	//The random number generator
	std::default_random_engine generator(static_cast<unsigned int>(time(NULL)));
	std::uniform_int_distribution<unsigned int> distribution;

	//Constants for hash table
	for (auto i = 0; i < num_hash_funcs; i++) {
		const auto new_a = distribution(generator);
		hash_constants.constants[i].x = (1 > new_a ? 1 : new_a);
		hash_constants.constants[i].y = distribution(generator);
	}

	//Constants for stash table
	stash_constants.x = distribution(generator);
	if (stash_constants.x == 0) stash_constants.x = 1;
	stash_constants.y = distribution(generator);
}


void hashing::build_hash_constants(uint2 & primary, uint2 & step)
{
	//The random number generator
	std::default_random_engine generator(static_cast<unsigned int>(time(NULL)));
	std::uniform_int_distribution<unsigned int> distribution;

	//First build it
	primary.x = distribution(generator);
	primary.y = distribution(generator);
	step.x = distribution(generator);
	step.y = distribution(generator);

	//Make it key dependent
	if (primary.x == 0) primary.x = 1;
	if (step.x == 0) step.x = 1;
}

int hashing::max_insert_attempts(const unsigned int num_entries, const unsigned int table_size)
{
	const auto lg_input_size = (float)(log((double)num_entries) / log(2.0));

	// Use an empirical formula for determining what the maximum number of
	// iterations should be.  Works OK in most situations.
	const auto load_factor = float(num_entries) / table_size;
	const auto ln_load_factor = (float)(log(load_factor) / log(2.71828183));

	const int max_iterations = (int)(4.0 * ceil(-1.0 / (0.028255 + 1.1594772 *
		ln_load_factor)* lg_input_size));
	//Can not be too small
	const int MAX_ITERATION_CONSTANT = 7;
	return std::max(max_iterations, MAX_ITERATION_CONSTANT);
}