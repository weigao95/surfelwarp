#pragma once
#include "common/logging.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
#include <cuda_runtime_api.h>

namespace surfelwarp {

	//For friend
	template<typename T>
	class DeviceSliceBufferArray;
	
	//The array slice class, provides non-owned
	//read-WRITE access to some array
	template<typename T>
	class DeviceArraySlice {
	private:
		T* m_array;
		size_t m_array_size;
	public:
		//Default copy/assign/move/destruct
		__host__ __device__ DeviceArraySlice() : m_array(nullptr), m_array_size(0) {}
		__host__ __device__ DeviceArraySlice(T* dev_arr, size_t size) : m_array(dev_arr), m_array_size(size) {}
		__host__ __device__ DeviceArraySlice(T* arr, size_t start, size_t end) {
			m_array_size = end - start;
			m_array = arr + start;
		}
		explicit __host__ DeviceArraySlice(const DeviceArray<T>& arr) : m_array((T*)arr.ptr()), m_array_size(arr.size()) {}

		//Simple interface
		__host__ __device__ __forceinline__ size_t Size() const { return m_array_size; }
		__host__ __device__ __forceinline__ size_t ByteSize() const { return m_array_size * sizeof(T); }
		__host__ __device__ __forceinline__ const T* RawPtr() const { return m_array; }
		__host__ __device__ __forceinline__ T* RawPtr() { return m_array; }
		__host__ __device__ DeviceArrayView<T> ArrayView() const { return DeviceArrayView<T>(m_array, m_array_size); }

		//Implicit convertor
		operator T*() { return m_array; }
		operator const T*() const { return m_array; }

		//The accessing method can only be processed on device
		__device__ __forceinline__ const T& operator[](size_t index) const { return m_array[index]; }
		__device__ __forceinline__ T& operator[](size_t index) { return m_array[index]; }

		//Download to std::vector
		void SyncToHost(std::vector<T>& h_vec, cudaStream_t stream = 0) const {
			h_vec.resize(m_array_size);
			cudaSafeCall(cudaMemcpyAsync(
				h_vec.data(),
				m_array,
				ByteSize(),
				cudaMemcpyDeviceToHost,
				stream
			));
		}
		
		//Upload from host
		void SyncFromHost(const std::vector<T>& h_vec, cudaStream_t stream = 0) {
			SURFELWARP_CHECK_EQ(h_vec.size(), m_array_size);
			cudaSafeCall(cudaMemcpyAsync(
				m_array,
				h_vec.data(),
				sizeof(T) * h_vec.size(),
				cudaMemcpyHostToDevice,
				stream
			));
		}
		
		//array_size modified by buffer array
		friend class DeviceSliceBufferArray<T>;
	};

	template<typename T>
	class DeviceSliceBufferArray {
	public:
		//Default copy/assign/move/delete
		__host__ __device__ DeviceSliceBufferArray(): m_buffer(), m_array() {}
		__host__ __device__ DeviceSliceBufferArray(T* buffer, size_t capacity)
		: m_buffer(buffer, capacity), m_array(buffer, 0) {}
		
		//The contructor on host will check the size
		__host__ DeviceSliceBufferArray(T* buffer, size_t capacity, size_t array_size)
			: m_buffer(buffer, capacity), m_array(buffer, array_size
		) {
			//Check the size
			if(array_size > capacity) {
				LOG(FATAL) << "The provided buffer is not enough";
			}
		}
		__host__ DeviceSliceBufferArray(const surfelwarp::DeviceArray<T>& arr)
			: m_buffer(arr), m_array((T*)arr.ptr(), 0) {}
		
		//Change the size of array
		__host__ void ResizeArrayOrException(size_t size) {
			if(size > m_buffer.Size()) {
				//Kill it
				LOG(FATAL) << "The provided buffer is not enough";
			}

			//Safe to resize the array
			m_array.m_array_size = size;
		}

		//Interface: no need to make it on device
		__host__ __forceinline__ size_t Capacity() const { return m_buffer.Size(); }
		__host__ __forceinline__ size_t BufferSize() const { return m_buffer.Size(); }
		__host__ __forceinline__ size_t ArraySize() const { return m_array.Size(); }
		__host__ __forceinline__ const T* Ptr() const { return m_buffer.RawPtr(); }
		__host__ __forceinline__ T* Ptr() { return m_buffer.RawPtr(); }
		__host__ __forceinline__ DeviceArraySlice<T> ArraySlice() const { return m_array; }
		__host__ __forceinline__ DeviceArrayView<T> ArrayReadOnly() const { return DeviceArrayView<T>(Ptr(), ArraySize()); }
		__host__ __forceinline__ DeviceArrayView<T> ArrayView() const { return DeviceArrayView<T>(Ptr(), ArraySize()); }
		
		//To make new slice from this one
		__host__ DeviceSliceBufferArray<T> BufferArraySlice(size_t start, size_t end) const {
			if(start > end || end > m_buffer.Size()) {
				LOG(FATAL) << "Incorrect slice size";
			}
			
			//Return inside slice
			return DeviceSliceBufferArray<T>((T*)m_buffer.RawPtr() + start, end - start);
		}
	private:
		DeviceArraySlice<T> m_buffer;
		DeviceArraySlice<T> m_array;
	};
}