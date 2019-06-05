/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef DEVICE_ARRAY_IMPL_HPP_
#define DEVICE_ARRAY_IMPL_HPP_


/////////////////////  Inline implementations of DeviceArrayPCL ////////////////////////////////////////////

template<class T> inline DeviceArrayPCL<T>::DeviceArrayPCL() {}
template<class T> inline DeviceArrayPCL<T>::DeviceArrayPCL(size_t size) : DeviceMemory(size * elem_size) {}
template<class T> inline DeviceArrayPCL<T>::DeviceArrayPCL(T *ptr, size_t size) : DeviceMemory(ptr, size * elem_size) {}
template<class T> inline DeviceArrayPCL<T>::DeviceArrayPCL(const DeviceArrayPCL& other) : DeviceMemory(other) {}
template<class T> inline DeviceArrayPCL<T>& DeviceArrayPCL<T>::operator=(const DeviceArrayPCL& other)
{ DeviceMemory::operator=(other); return *this; }

template<class T> inline void DeviceArrayPCL<T>::create(size_t size)
{ DeviceMemory::create(size * elem_size); }
template<class T> inline void DeviceArrayPCL<T>::release()
{ DeviceMemory::release(); }

template<class T> inline void DeviceArrayPCL<T>::copyTo(DeviceArrayPCL& other) const
{ DeviceMemory::copyTo(other); }
template<class T> inline void DeviceArrayPCL<T>::upload(const T *host_ptr, size_t size)
{ DeviceMemory::upload(host_ptr, size * elem_size); }
template<class T> inline void DeviceArrayPCL<T>::download(T *host_ptr) const
{ DeviceMemory::download( host_ptr ); }

template<class T> void DeviceArrayPCL<T>::swap(DeviceArrayPCL& other_arg) { DeviceMemory::swap(other_arg); }

template<class T> inline DeviceArrayPCL<T>::operator T*() { return ptr(); }
template<class T> inline DeviceArrayPCL<T>::operator const T*() const { return ptr(); }
template<class T> inline size_t DeviceArrayPCL<T>::size() const { return sizeBytes() / elem_size; }

template<class T> inline       T* DeviceArrayPCL<T>::ptr()       { return DeviceMemory::ptr<T>(); }
template<class T> inline const T* DeviceArrayPCL<T>::ptr() const { return DeviceMemory::ptr<T>(); }

template<class T> template<class A> inline void DeviceArrayPCL<T>::upload(const std::vector<T, A>& data) { upload(&data[0], data.size()); }
template<class T> template<class A> inline void DeviceArrayPCL<T>::download(std::vector<T, A>& data) const { data.resize(size()); if (!data.empty()) download(&data[0]); }

/////////////////////  Inline implementations of DeviceArray2DPCL ////////////////////////////////////////////

template<class T> inline DeviceArray2DPCL<T>::DeviceArray2DPCL() {}
template<class T> inline DeviceArray2DPCL<T>::DeviceArray2DPCL(int rows, int cols) : DeviceMemory2D(rows, cols * elem_size) {}
template<class T> inline DeviceArray2DPCL<T>::DeviceArray2DPCL(int rows, int cols, void *data, size_t stepBytes) : DeviceMemory2D(rows, cols * elem_size, data, stepBytes) {}
template<class T> inline DeviceArray2DPCL<T>::DeviceArray2DPCL(const DeviceArray2DPCL& other) : DeviceMemory2D(other) {}
template<class T> inline DeviceArray2DPCL<T>& DeviceArray2DPCL<T>::operator=(const DeviceArray2DPCL& other)
{ DeviceMemory2D::operator=(other); return *this; }

template<class T> inline void DeviceArray2DPCL<T>::create(int rows, int cols)
{ DeviceMemory2D::create(rows, cols * elem_size); }
template<class T> inline void DeviceArray2DPCL<T>::release()
{ DeviceMemory2D::release(); }

template<class T> inline void DeviceArray2DPCL<T>::copyTo(DeviceArray2DPCL& other) const
{ DeviceMemory2D::copyTo(other); }
template<class T> inline void DeviceArray2DPCL<T>::upload(const void *host_ptr, size_t host_step, int rows, int cols)
{ DeviceMemory2D::upload(host_ptr, host_step, rows, cols * elem_size); }
template<class T> inline void DeviceArray2DPCL<T>::download(void *host_ptr, size_t host_step) const
{ DeviceMemory2D::download( host_ptr, host_step ); }

template<class T> template<class A> inline void DeviceArray2DPCL<T>::upload(const std::vector<T, A>& data, int cols)
{ upload(&data[0], cols * elem_size, data.size()/cols, cols); }

template<class T> template<class A> inline void DeviceArray2DPCL<T>::download(std::vector<T, A>& data, int& elem_step) const
{ elem_step = cols(); data.resize(cols() * rows()); if (!data.empty()) download(&data[0], colsBytes());  }

template<class T> void  DeviceArray2DPCL<T>::swap(DeviceArray2DPCL& other_arg) { DeviceMemory2D::swap(other_arg); }

template<class T> inline       T* DeviceArray2DPCL<T>::ptr(int y)       { return DeviceMemory2D::ptr<T>(y); }
template<class T> inline const T* DeviceArray2DPCL<T>::ptr(int y) const { return DeviceMemory2D::ptr<T>(y); }
            
template<class T> inline DeviceArray2DPCL<T>::operator T*() { return ptr(); }
template<class T> inline DeviceArray2DPCL<T>::operator const T*() const { return ptr(); }

template<class T> inline int DeviceArray2DPCL<T>::cols() const { return DeviceMemory2D::colsBytes()/elem_size; }
template<class T> inline int DeviceArray2DPCL<T>::rows() const { return DeviceMemory2D::rows(); }

template<class T> inline size_t DeviceArray2DPCL<T>::elem_step() const { return DeviceMemory2D::step()/elem_size; }


#endif /* DEVICE_ARRAY_IMPL_HPP_ */
