#pragma once

//Disable cuda on Eigen
#ifndef EIGEN_NO_CUDA
#define EIGEN_NO_CUDA
#endif

//Perform debug sync and check cuda error
//#define CUDA_DEBUG_SYNC_CHECK

//Clip pixels at boundary for better windowed search. 
//Please refer to clip.md under imgproc directory for details
#ifndef boundary_clip
#define boundary_clip 20
#endif


//For pcl access of new in debug mode
#if defined(CUDA_DEBUG_SYNC_CHECK)
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#endif

#define CUDA_CHECKERR_SYNC 1

//The constants need to accessed on device
#define d_max_num_nodes 4096
#define d_node_radius 0.025f // [meter]
#define d_node_radius_square (d_node_radius * d_node_radius)

//The scale of fusion map, will be accessed on device
#define d_fusion_map_scale 4


//Normalize the interpolate weight to 1
#define USE_INTERPOLATE_WEIGHT_NORMALIZATION


//Use dense image density term in the solver
//#define USE_DENSE_IMAGE_DENSITY_TERM

//Fix boost broken issue with cuda compile
#ifdef __CUDACC__
#define BOOST_PP_VARIADICS 0
#endif
