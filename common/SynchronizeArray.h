//
// Created by wei on 3/16/18.
//

#pragma once
#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/DeviceBufferArray.h"

namespace surfelwarp {
	
	
	/**
	 * \brief The array with synchronize functionalities. Note that the content of the
	 *        array is not guarantee to be synchronized and need explict sync. However, 
	 *        the array size of host and device array are always the same.
	 * \tparam T 
	 */
	template<typename T>
	class SynchronizeArray {
	public:
		explicit SynchronizeArray() : m_host_array(), m_device_array() {}
		explicit SynchronizeArray(size_t capacity) {
			AllocateBuffer(capacity);
		}
		~SynchronizeArray() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(SynchronizeArray);
		
		//The accessing interface
		size_t Capacity() const { return m_device_array.Capacity(); }
		size_t DeviceArraySize() const { return m_device_array.ArraySize(); }
		size_t HostArraySize() const { return m_host_array.size(); }
		
		DeviceArrayView<T> DeviceArrayReadOnly() const { return DeviceArrayView<T>(m_device_array.Array()); }
		surfelwarp::DeviceArray<T> DeviceArray() const { return m_device_array.Array(); }
		DeviceArraySlice<T> DeviceArrayReadWrite() { return m_device_array.ArraySlice(); }
		
		std::vector<T>& HostArray() { return m_host_array; }
		const std::vector<T>& HostArray() const { return m_host_array; }

		//Access the raw pointer
		const T* DevicePtr() const { return m_device_array.Ptr(); }
		T* DevicePtr() { return m_device_array.Ptr(); }
		
		//The (possible) allocate interface
		void AllocateBuffer(size_t capacity) {
			m_host_array.reserve(capacity);
			m_device_array.AllocateBuffer(capacity);
		}

		//the DeviceBufferArray has implement resize with copy
		bool ResizeArray(size_t size, bool allocate = false) {
			if(m_device_array.ResizeArray(size, allocate) == true) {
				m_host_array.resize(size);
				return true;
			}
			
			//The device array can not resize success
			//The host and device are in the same size
			return false;
		}
		void ResizeArrayOrException(size_t size) {
			m_device_array.ResizeArrayOrException(size);
			m_host_array.resize(size);
		}
		
		//Clear the array of both host and device array
		//But DO NOT TOUCH the allocated buffer
		void ClearArray() {
			ResizeArray(0);
		}
		
		//The sync interface
		void SynchronizeToDevice(cudaStream_t stream = 0) {
			//Update the size
			m_device_array.ResizeArrayOrException(m_host_array.size());
			
			//Actual sync
			cudaSafeCall(cudaMemcpyAsync(
				m_device_array.Ptr(),
				m_host_array.data(), 
				sizeof(T) * m_host_array.size(),
				cudaMemcpyHostToDevice, stream
			));
		}
		void SynchronizeToHost(cudaStream_t stream = 0, bool sync = true) {
			//Resize host array
			m_host_array.resize(m_device_array.ArraySize());
			
			//Actual sync
			cudaSafeCall(cudaMemcpyAsync(
				m_host_array.data(), 
				m_device_array.Ptr(),
				sizeof(T) * m_host_array.size(), 
				cudaMemcpyDeviceToHost, stream
			));
			
			if(sync) {
				//Before using on host, must call stream sync
				//But the call might be delayed
				cudaSafeCall(cudaStreamSynchronize(stream));
			}
		}
	
	private:
		std::vector<T> m_host_array;
		DeviceBufferArray<T> m_device_array;
	};
}
