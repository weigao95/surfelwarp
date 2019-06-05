//
// Created by wei on 2/16/18.
//

#pragma once

//Whether use blocking in hashing
#define COMPACTION_USE_BLOCKING
//#define COMPACTION_COUNT_BLOCKING

namespace hashing {
    //Parameters for host-accessed hash table
    const int max_restart_attempts = 10;
    const int num_hash_funcs = 4;
    const int stash_table_size = 647;
	const int insert_thread_block = 64;
    const float space_factor = 1.4f;
	const unsigned max_probe_attempt = 1000;
}
