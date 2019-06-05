#pragma once

#include "imgproc/segmentation/permutohedral_common.h"

template<int FeatureDim>
float surfelwarp::permutohedral_scale_factor(const int index) {
	return (FeatureDim + 1) * sqrtf((1.0f / 6.0f) / ((index + 1) * (index + 2)));
}

template<int FeatureDim>
unsigned surfelwarp::LatticeCoordKey<FeatureDim>::hash() const
{
	unsigned hash_value = 0;
	for(auto i = 0; i < FeatureDim; i++) {
		hash_value += key[i];
		//hash_value *= 1664525;
		hash_value *= 1500007;
	}
	return hash_value;
}

template<int FeatureDim>
char surfelwarp::LatticeCoordKey<FeatureDim>::less_than(const LatticeCoordKey<FeatureDim>& rhs) const
{
	char is_less_than = 0;
	for(auto i = 0; i < FeatureDim; i++) {
		if(key[i] < rhs.key[i]) {
			is_less_than = 1;
			break;
		} else if(key[i] > rhs.key[i]) {
			is_less_than = -1;
			break;
		}
		//Else, continue
	}
	return is_less_than;
}


template<int FeatureDim>
void surfelwarp::LatticeCoordKey<FeatureDim>::set_null()
{
	for(auto i = 0; i < FeatureDim; i++) {
		key[i] = 1 << 14;
	}
}

template<int FeatureDim>
bool surfelwarp::LatticeCoordKey<FeatureDim>::is_null() const {
	bool null_key = true;
	for(auto i = 0; i < FeatureDim; i++) {
		if(key[i] != (1 << 14)) null_key = false;
	}
	return null_key;
}


template<int FeatureDim>
void surfelwarp::permutohedral_lattice(
	const float* feature,
	LatticeCoordKey<FeatureDim>* lattice_coord_keys,
	float* barycentric
) {
	float elevated[FeatureDim + 1];
	elevated[FeatureDim] = -FeatureDim * (feature[FeatureDim - 1]) * permutohedral_scale_factor<FeatureDim>(FeatureDim - 1);
	for (int i = FeatureDim - 1; i > 0; i--) {
		elevated[i] = (elevated[i + 1] -
		                 i * (feature[i - 1]) * permutohedral_scale_factor<FeatureDim>(i - 1) +
		                 (i + 2) * (feature[i]) * permutohedral_scale_factor<FeatureDim>(i));
	}
	elevated[0] = elevated[1] + 2 * (feature[0]) * permutohedral_scale_factor<FeatureDim>(0);
	
	short greedy[FeatureDim + 1];
	signed short sum = 0;
	for (int i = 0; i <= FeatureDim; i++) {
		float v = elevated[i] * (1.0f / (FeatureDim + 1));
		float up = ceilf(v) * (FeatureDim + 1);
		float down = floorf(v) * (FeatureDim + 1);
		if (up - elevated[i] < elevated[i] - down) {
			greedy[i] = (signed short) up;
		} else {
			greedy[i] = (signed short) down;
		}
		sum += greedy[i];
	}
	sum /= FeatureDim + 1;
	
	//Sort differential to find the permutation between this simplex and the canonical one
	short rank[FeatureDim + 1];
	for (int i = 0; i <= FeatureDim; i++) {
		rank[i] = 0;
		for (int j = 0; j <= FeatureDim; j++) {
			if (elevated[i] - greedy[i] < elevated[j] - greedy[j] ||
			    (elevated[i] - greedy[i] == elevated[j] - greedy[j]
			     && i > j)) {
				rank[i]++;
			}
		}
	}
	
	//Sum too large, need to bring down the ones with the smallest differential
	if (sum > 0) {
		for (int i = 0; i <= FeatureDim; i++) {
			if (rank[i] >= FeatureDim + 1 - sum) {
				greedy[i] -= FeatureDim + 1;
				rank[i] += sum - (FeatureDim + 1);
			} else {
				rank[i] += sum;
			}
		}
	} else if (sum < 0) { //Sum too small, need to bring up the ones with largest differential
		for (int i = 0; i <= FeatureDim; i++) {
			if (rank[i] < -sum) {
				greedy[i] += FeatureDim + 1;
				rank[i] += (FeatureDim + 1) + sum;
			} else {
				rank[i] += sum;
			}
		}
	}
	
	
	//Turn delta into barycentric coords
	for (int i = 0; i <= FeatureDim + 1; i++) {
		barycentric[i] = 0;
	}
	
	for (int i = 0; i <= FeatureDim; i++) {
		float delta = (elevated[i] - greedy[i]) * (1.0f / (FeatureDim + 1));
		barycentric[FeatureDim - rank[i]] += delta;
		barycentric[FeatureDim + 1 - rank[i]] -= delta;
	}
	barycentric[0] += 1.0f + barycentric[FeatureDim + 1];
	
	//Construct the key and their weight
	for (auto color = 0; color <= FeatureDim; color++) {
		//Compute the location of the lattice point explicitly (all but
		//the last coordinate - it's redundant because they sum to zero)
		short* key = lattice_coord_keys[color].key;
		for (int i = 0; i < FeatureDim; i++) {
			key[i] = greedy[i] + color;
			if (rank[i] > FeatureDim - color) key[i] -= (FeatureDim + 1);
		}
	}
}