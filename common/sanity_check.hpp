#pragma once
#include "common/sanity_check.h"
#include <iostream>

template<typename T>
bool surfelwarp::isElementUniqueNaive(const std::vector<T> &vec, const T empty) {
	for(auto i = 0; i < vec.size(); i++) {
		const T& elem_i = vec[i];
		if(elem_i == empty) continue;
		for(auto j = 0; j < vec.size(); j++) {
			if(i == j) continue;
			const T& elem_j = vec[j];
			if(elem_j == empty) continue;

			//Duplicate
			if((elem_j == elem_i)) {
				return false;
			}
		}
	}
	return true;
}

template <typename T>
bool surfelwarp::isElementUniqueNonEmptyNaive(const std::vector<T>& vec, const T empty)
{
	for(auto i = 0; i < vec.size(); i++) {
		const T& elem_i = vec[i];
		if(elem_i == empty) {
			std::cout << "Empty element!!" << std::endl;
			return false;
		}

		for(auto j = 0; j < vec.size(); j++) {
			if(i == j) continue;
			const T& elem_j = vec[j];
			if(elem_j == empty) continue;

			//Duplicate
			if((elem_j == elem_i)) {
				return false;
			}
		}
	}
	return true;
}

template<typename T>
double surfelwarp::averageDuplicate(const std::vector<T> &vec, const T empty) {
	std::vector<int> duplicate;
	duplicate.resize(vec.size());
	
	//Count the duplicate loop
	for(auto i = 0; i < vec.size(); i++) {
		const T& elem_i = vec[i];
		if(elem_i == empty) {
			duplicate[i] = 0;
			continue;
		}
		
		//Count self
		duplicate[i] = 1;
		
		for(auto j = 0; j < vec.size(); j++) {
			if(i == j) continue;
			const T& elem_j = vec[j];
			if(elem_j == empty) continue;
			
			//Duplicate
			if((elem_j == elem_i)) {
				duplicate[i]++;
			}
		}
	}
	
	double total_duplicate = 0.0;
	int valid_count = 0;
	for(auto i = 0; i < duplicate.size(); i++) {
		if(vec[i] == empty) continue;
		valid_count++;
		total_duplicate += duplicate[i];
	}
	
	return (total_duplicate / valid_count);
}
