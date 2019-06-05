//
// Created by wei on 2/13/18.
//

#pragma once


//The bin size of Bin Blocked CSR format
//Should always be the warp size
#define bin_size 32

//The maximum number of nodes considered in this application
#define max_num_nodes 4096

//The number of elements that in a block
//can handle
#define reduce_block_warps 32
#define reduce_block_threads (32 * reduce_block_warps)

//The number of rows that a thread block is able to deal with
#define reduce_block_rows reduce_block_threads

//The maximum matrix size (32768) to perform reduce in combined kernel
//For larger matrix, the reduction when computing dot(vec_0, vec_1)
//cannot be implemented in combined kernels
//#define max_matrix_size (reduce_block_warps * reduce_block_rows)

//The number of required reduct blocks
#define num_reduce_blocks_6x6 ((6 * max_num_nodes) / reduce_block_rows)
#define max_reduce_blocks 32

