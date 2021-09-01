//
// Created by wei on 3/16/18.
//

#pragma once
#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/logging.h"
#include "common/ArrayView.h"
#include "common/ArraySlice.h"

namespace surfelwarp {
	

	template<typename T>
	class DeviceBufferArray {
	public:
		explicit DeviceBufferArray() : m_buffer(nullptr, 0), m_array(nullptr, 0) {}
		explicit DeviceBufferArray(size_t capacity) {
			AllocateBuffer(capacity);
			m_array = DeviceArray<T>(m_buffer.ptr(), 0);
		}
		~DeviceBufferArray() = default;

		//No implicit copy/assign/move
		SURFELWARP_NO_COPY_ASSIGN_MOVE(DeviceBufferArray);

		//Accessing method
		DeviceArray<T> Array() const { return m_array; }
		DeviceArrayView<T> ArrayView() const { return DeviceArrayView<T>(m_array.ptr(), m_array.size()); }
		DeviceArrayView<T> ArrayReadOnly() const { return DeviceArrayView<T>(m_array.ptr(), m_array.size()); }
		DeviceArraySlice<T> ArraySlice() { return DeviceArraySlice<T>(m_array.ptr(), m_array.size()); }
		DeviceArray<T> Buffer() const { return m_buffer; }
		
		//The swap method
		void swap(DeviceBufferArray<float>& other) {
			m_buffer.swap(other.m_buffer);
			m_array.swap(other.m_array);
		}
		
		//Cast to raw pointer
		const T* Ptr() const { return m_buffer.ptr(); }
		T* Ptr() { return m_buffer.ptr(); }
		operator T*() { return m_buffer.ptr(); }
		operator const T*() const { return m_buffer.ptr(); }
		
		//Query the size
		size_t Capacity() const { return m_buffer.size(); }
		size_t BufferSize() const { return m_buffer.size(); }
		size_t ArraySize() const { return m_array.size(); }

		//The allocation and changing method
		void AllocateBuffer(size_t capacity) {
			if(m_buffer.size() > capacity) return;
			m_buffer.create(capacity);
			m_array = DeviceArray<T>(m_buffer.ptr(), 0);
		}
		void ReleaseBuffer() {
			if(m_buffer.size() > 0) m_buffer.release();
		}
		
		bool ResizeArray(size_t size, bool allocate = false) {
			if(size <= m_buffer.size()) {
				m_array = DeviceArray<T>(m_buffer.ptr(), size);
				return true;
			} 
			else if(allocate) {
				const size_t prev_size = m_array.size();

				//Need to copy the old elements
				DeviceArray<T> old_buffer = m_buffer;
				m_buffer.create(static_cast<size_t>(size * 1.5));
				if(prev_size > 0) {
					cudaSafeCall(cudaMemcpy(m_buffer.ptr(), old_buffer.ptr(), sizeof(T) * prev_size, cudaMemcpyDeviceToDevice));
					old_buffer.release();
				}

				//Correct the size
				m_array = DeviceArray<T>(m_buffer.ptr(), size);
				return true;
			} 
			else {
				return false;
			}
		}

		
		void ResizeArrayOrException(size_t size) {
			if (size > m_buffer.size()) {
				LOG(FATAL) << "The pre-allocated buffer is not enough";
			}

			//Change the size of array
			m_array = DeviceArray<T>(m_buffer.ptr(), size);
		}
	private:
		DeviceArray<T> m_buffer;
		DeviceArray<T> m_array;
	};


}
