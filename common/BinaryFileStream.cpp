#include "common/BinaryFileStream.h"

surfelwarp::BinaryFileStream::BinaryFileStream(const char * path, Mode mode)
{
	std::FILE* file_ptr = nullptr;
	if(mode == Mode::ReadOnly) {
		file_ptr = std::fopen(path, "rb");
	} 
	else if(mode == Mode::WriteOnly) {
		file_ptr = std::fopen(path, "wb");
	}

	//Update to the pointer
	m_file_handle = file_ptr;
}

surfelwarp::BinaryFileStream::~BinaryFileStream()
{
	std::fclose(m_file_handle);
	m_file_handle = nullptr;
}

size_t surfelwarp::BinaryFileStream::Read(void * ptr, size_t bytes)
{
	return std::fread(ptr, 1, bytes, m_file_handle);
}

bool surfelwarp::BinaryFileStream::Write(const void * ptr, size_t bytes)
{
	return (std::fwrite(ptr, 1, bytes, m_file_handle) == bytes);
}