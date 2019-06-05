#pragma once

#include "hashing/hash_common.h"
#include "hashing/hash_config.h"

namespace hashing {
    //Prime number larger than the largest practical hash table size
    const unsigned int prime_divisor = 4294967291u;

    //The hash functions used for all types of tables
    __host__ __device__ __forceinline__
    unsigned int hash_value(const uint2& hash_constants, const KeyT key) {
        return ((hash_constants.x ^ key) + hash_constants.y) % prime_divisor;
    }

    //The stash table position
    __host__ __device__ __forceinline__
    unsigned int stash_position(const uint2& stash_constants, const unsigned int key) {
        return (stash_constants.x ^ key + stash_constants.y) % stash_table_size;
    }

    __host__ __device__ __forceinline__
    KeyT entry_key(const HashEntry entry) {
        return (KeyT)(entry >> 32);
    }

    __host__ __device__ __forceinline__
    ValueT entry_value(const HashEntry entry) {
        return (ValueT)(entry & 0xffffffff);
    }

    __host__ __device__ __forceinline__
    HashEntry make_entry(const KeyT key, const ValueT value) {
        return (HashEntry(key) << 32) + value;
    }

    __host__ __device__ __forceinline__
    int divUp(int a, int b) {
        return (a + b - 1) / b;
    }

	__host__ __device__ __forceinline__
	unsigned int next_location(
		const uint2 hash_constants[num_hash_funcs],
		const unsigned int table_size,
		const unsigned int prev_location,
		const KeyT key
	)
    {
		unsigned int locations[num_hash_funcs];
#pragma unroll
		for (auto i = 0; i < num_hash_funcs; i++) {
			locations[i] = hash_value(hash_constants[i], key) % table_size;
		}

		//Ok, figure out the next place
		auto next_location = locations[0];
		for (int i = num_hash_funcs - 2; i >= 0; i--) {
			next_location = (prev_location == locations[i] ? locations[i + 1] : next_location);
		}
		return next_location;
    }
}