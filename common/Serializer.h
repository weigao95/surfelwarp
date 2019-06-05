//
// Created by wei on 2/12/18.
//

#pragma once
#include <vector>
#include <iostream>
#include "common/logging.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
//#include "common/Stream.h"
#include "common/custom_type_traits.h"

namespace surfelwarp {
	
	/**
	 * \brief the general class handler for serialization
	 * \tparam T
	 */
	template<typename T>
	struct SerializeHandler;
	
	/**
	 * \brief the class to help to SerializeHandler
	 */
	template<bool condition, typename Then, typename Else, typename T>
	struct IfThenElse;
	
	template<typename Then, typename Else, typename T>
	struct IfThenElse<true, Then, Else, T> {
		inline static void Write(Stream* stream, const T& object) {
			Then::Write(stream, object);
		}
		inline static bool Read(Stream* stream, T* object) {
			return Then::Read(stream, object);
		}
	};
	
	template<typename Then, typename Else, typename T>
	struct IfThenElse<false, Then, Else, T> {
		inline static void Write(Stream* stream, const T& object) {
			Else::Write(stream, object);
		}
		inline static bool Read(Stream* stream, T* object) {
			return Else::Read(stream, object);
		}
	};

	/**
	 * \brief The handler for plain-old-data
	 * \tparam T The type to serialize
	 */
	template<typename T>
	struct PODSerializeHandler {
		inline static void Write(Stream* stream, const T& data) {
			stream->Write(&data, sizeof(T));
		}
		inline static bool Read(Stream* stream, T* data) {
			return stream->Read(data, sizeof(T));
		}
	};


	/**
	 * \brief The handler for class which implements streamSave/Load
	 */
	template<typename T>
	struct OutClassSaveLoadSerializeHandler {
		inline static void Write(Stream* stream, const T& data) {
			streamSave(stream, data);
		}
		inline static bool Read(Stream* stream, T* data) {
			return streamLoad(stream, data);
		}
	};
	
	
	/**
	 * \brief The handler for class with explicit Save and Load
	 * \tparam T The class should implement Save and Load method
	 */
	template<typename T>
	struct InClassSaveLoadSerializeHandler {
		inline static void Write(Stream* stream, const T& object) {
			object.Save(stream);
		}
		inline static bool Read(Stream* stream, T* object) {
			return object->Load(stream);
		}
	};

	/**
	 * \brief the handler for pod vector
	 * \tparam T the value type of the vector
	 */
	template<typename T>
	struct PODVectorSerializeHandler {
		inline static void Write(Stream* stream, const std::vector<T>& vec) {
			uint64_t vector_size = static_cast<uint64_t>(vec.size());
			stream->Write(&vector_size, sizeof(vector_size));
			if(vector_size != 0) {
				stream->Write(&vec[0], sizeof(T) * vec.size());
			}
		}

		inline static bool Read(Stream* stream, std::vector<T>* vec) {
			uint64_t raw_vec_size;
			//Read the size and check the read is success
			if(!(stream->Read(&raw_vec_size, sizeof(uint64_t)))) {
				return false;
			}

			//Expect the input is non-empty, but allocate one here?
			if(vec == nullptr) {
				vec = new std::vector<T>();
			}

			//Reserve the space
			size_t vec_size = static_cast<size_t>(raw_vec_size);
			vec->resize(vec_size);

			//Read the actual data
			if(raw_vec_size != 0) {
				return stream->Read(vec->data(), sizeof(T) * vec_size);
			} else {
				return true;
			}
		}
	};// the handler for pod vector
	

	template<typename T>
	struct ComposedVectorSerializeHandler
	{
		inline static void Write(Stream* stream, const std::vector<T>& vec) {
			uint64_t vector_size = static_cast<uint64_t>(vec.size());
			stream->Write(&vector_size, sizeof(vector_size));
			if(vector_size == 0) return;
			//Need to use the customized handler
			for(auto i = 0; i < vec.size(); i++) {
				SerializeHandler<T>::Write(stream, vec[i]);
			}
		}
		
		inline static bool Read(Stream* stream, std::vector<T>* vec) {
			uint64_t raw_vec_size;
			//Read the size and check the read is success
			if(!(stream->Read(&raw_vec_size, sizeof(uint64_t)))) {
				return false;
			}
			
			//Expect the input is non-empty, but allocate one here?
			if(vec == nullptr) {
				vec = new std::vector<T>();
			}
			
			//Reserve the space
			size_t vec_size = static_cast<size_t>(raw_vec_size);
			vec->resize(vec_size);
			
			//Check if this is an empty vector
			if(vec->size() == 0) return true;
			
			//Load the element for each element
			for(auto i = 0; i < vec->size(); i++) {
				SerializeHandler<T>::Read(stream, &((*vec)[i]));
			}
		}
	};
	
	//The handler for non-container/compose type
	template<typename T>
	struct ExplicitSaveLoadHandler {
		inline static void Write(Stream* stream, const T& object) {
			IfThenElse<
				has_inclass_saveload<T>::value,
				InClassSaveLoadSerializeHandler<T>,
				OutClassSaveLoadSerializeHandler<T>,
				T
			>::Write(stream, object);
		}
		
		inline static bool Read(Stream* stream, T* object) {
			return IfThenElse<
				has_inclass_saveload<T>::value,
				InClassSaveLoadSerializeHandler<T>,
				OutClassSaveLoadSerializeHandler<T>,
				T
			>::Read(stream, object);
		}
	};
	
	//The handler for non-container types
	template<typename T>
	struct SerializeHandler {
		//The write interface
		inline static void Write(Stream* stream, const T& object) {
			IfThenElse<
				is_pod<T>::value,
				PODSerializeHandler<T>,
				ExplicitSaveLoadHandler<T>,
				T
			>::Write(stream, object);
		}
		
		//The read interface
		inline static bool Read(Stream* stream, T* object) {
			return IfThenElse<
				is_pod<T>::value,
				PODSerializeHandler<T>,
				ExplicitSaveLoadHandler<T>,
				T
			>::Read(stream, object);
		}
	};
	
	//The handler for vector container types
	template<typename T>
	struct SerializeHandler<std::vector<T>> {
		//The write interface
		inline static void Write(Stream* stream, const std::vector<T>& vec) {
			IfThenElse<is_pod<T>::value,
				PODVectorSerializeHandler<T>,
				ComposedVectorSerializeHandler<T>,
				std::vector<T>
			>::Write(stream, vec);
		}
		
		//The read interface
		inline static bool Read(Stream* stream, std::vector<T>* vec) {
			return IfThenElse<
				is_pod<T>::value,
				PODVectorSerializeHandler<T>,
				ComposedVectorSerializeHandler<T>,
				std::vector<T>
			>::Read(stream, vec);
		}
	};


	//The handler for device access vector
	template <typename T>
	struct SerializeHandler<DeviceArray<T>> {
		inline static void Write(Stream* stream, const DeviceArray<T>& vec) {
			std::vector<T> h_vec;
			vec.download(h_vec);
			SerializeHandler<std::vector<T>>::Write(stream, h_vec);
		}

		inline static bool Read(Stream* stream, DeviceArray<T>* vec) {
			std::vector<T> h_vec;
			const bool read_success = SerializeHandler<std::vector<T>>::Read(stream, &h_vec);
			if(!read_success) return false;

			//Read is success, upload it to device
			vec->upload(h_vec);
			return true;
		}
	};
	
	
	template<typename T>
	struct SerializeHandler<DeviceArrayView<T>> {
		inline static void Write(Stream* stream, const DeviceArrayView<T>& vec) {
			std::vector<T> h_vec;
			vec.Download(h_vec);
			SerializeHandler<std::vector<T>>::Write(stream, h_vec);
		}
		
		inline static bool Read(Stream* stream, DeviceArrayView<T>* vec) {
			LOG(ERROR) << "Can not read into Read-Only ArrayView. Please load into std::vector or DeviceArray";
			return false;
		}
	};

}// namespace surfelwarp
