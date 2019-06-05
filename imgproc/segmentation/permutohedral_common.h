//
// Created by wei on 2/25/18.
//

#pragma once

#include "common/common_types.h"


namespace surfelwarp {
	
	template<int FeatureDim = 5>
	__host__ __device__ __forceinline__
	float permutohedral_scale_factor(const int index);
	
	//Small struct to hold the lattice coordinate as key
	template<int FeatureDim = 5>
	struct LatticeCoordKey {
		//Only maintain the first FeatureDim elements.
		//As the total sum to zero.
		short key[FeatureDim];

		//The hashing of this key
		__host__ __device__ __forceinline__ unsigned hash() const;

		
		/**
		 * \brief The comparator of the key
		 * \param rhs 
		 * \return 1 if truely less than, -1 is truely larger than, 0 equals
		 */
		__host__ __device__ __forceinline__ char less_than(const LatticeCoordKey<FeatureDim>& rhs) const;

		//Make this a null ptr
		__host__ __device__ __forceinline__ void set_null();
		__host__ __device__ __forceinline__ bool is_null() const;

		//Operator
		__host__ __device__ __forceinline__ bool operator==(const LatticeCoordKey<FeatureDim>& rhs) const
		{
			bool equal = true;
			for(auto i = 0; i < FeatureDim; i++) {
				if(key[i] != rhs.key[i]) equal = false;
			}
			return equal;
		}
	};
	
	

	/**
	 * \brief Compute the lattice key and the weight of the lattice point
	 *        surround this feature. 
	 * \tparam FeatureDim 
	 * \param feature The feature vector, in the size of FeatureDim
	 * \param lattice_coord_keys The lattice coord keys nearby this feature. The
	 *                           array is in the size of FeatureDim + 1.
	 * \param barycentric The weight value, in the size of FeatureDim + 2, while
	 *                    the first FeatureDim + 1 elements match the weight
	 *                    of the lattice_coord_keys
	 */
	template<int FeatureDim = 5>
	__host__ __device__ __forceinline__
	void permutohedral_lattice(
		const float* feature,
		LatticeCoordKey<FeatureDim>* lattice_coord_keys,
		float* barycentric
	);
}


//The implementation file
#include "imgproc/segmentation/permutohedral_common.hpp"