#pragma once
#include "common/Stream.h"
#include "common/Serializer.h"

template<typename T>
inline bool surfelwarp::Stream::SerializeRead(T * output)
{
	return SerializeHandler<T>::Read(this, output);
}

template<typename T>
inline void surfelwarp::Stream::SerializeWrite(const T & object)
{
	SerializeHandler<T>::Write(this, object);
}