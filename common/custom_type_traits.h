//
// Created by wei on 3/8/18.
//

#pragma once

#include <type_traits>
#include <string>
#include <exception>

namespace surfelwarp {
	
	template <typename T>
	struct is_pod {
		static const bool value = std::is_pod<T>::value;
	};
	
	//Forward declaration
	class Stream;
	
	//The generic function for save and load from stream
	template <typename T>
	inline bool streamLoad(Stream* stream, T* object) {
		throw new std::runtime_error("The stream load function is not implemented");
	}
	
	template <typename T>
	inline void streamSave(Stream* stream, const T& object) {
		throw new std::runtime_error("The stream save function is not implemented");
	}
	
	template <typename T>
	struct has_outclass_saveload {
		static const bool value = false;
	};
	
	template<typename T>
	struct has_inclass_saveload {
		static const bool value = false;
	};
}


