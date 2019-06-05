//
// Created by wei on 2/16/18.
//

#pragma once

#include "hashing/hash_config.h"
#include <vector_functions.h>
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace hashing {

    //The struct in hash table
    using HashEntry = unsigned long long;
    using KeyT = unsigned int;
    using ValueT = unsigned int;

    //The struct to transfer the data to kernels
    struct HashConstants {
        uint2 constants[num_hash_funcs];
    };

    //The predefined-entry for the table
    const KeyT EmptyKey = 0xffffffffu;
    const KeyT KeyNotFound = 0xffffffffu;
	const unsigned EmptyTicket = 0xffffffffu;
	const unsigned InvalidIndex = 0xffffffffu;
    const HashEntry EmptyEntry = HashEntry(EmptyKey) << 32;

	//Compute the hash constants and attempt insertations
	void build_hash_constants(HashConstants& hash_constants, uint2& stash_constants);
	void build_hash_constants(uint2& primary, uint2& step);
	int max_insert_attempts(const unsigned int num_entries, const unsigned int table_size);

    //The safe call macro
#if !defined(cudaSafeCall)
#define cudaSafeCall( call) {                                                \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(1);                                                             \
    } }
#endif
}
