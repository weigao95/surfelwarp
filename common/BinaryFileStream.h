#pragma once
#include "common/Stream.h"
#include <memory>

namespace surfelwarp {
	
	class BinaryFileStream : public Stream {
	private:
		std::FILE* m_file_handle;
	public:
		using Ptr = std::shared_ptr<BinaryFileStream>;

		enum class Mode {
			ReadOnly,
			WriteOnly
		};

		//Constructor and destructor
		explicit BinaryFileStream(const char* path, Mode mode = Mode::ReadOnly);
		~BinaryFileStream() override;
		
		//No copy/assgin
		BinaryFileStream(const BinaryFileStream&) = delete;
		BinaryFileStream(BinaryFileStream&&) = delete;
		BinaryFileStream& operator=(const BinaryFileStream&) = delete;
		BinaryFileStream& operator=(BinaryFileStream&&) = delete;

		//The main interface
		size_t Read(void* ptr, size_t bytes) override;
		bool Write(const void* ptr, size_t bytes) override;
	};

}
