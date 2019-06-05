//
// Created by wei on 3/16/18.
//

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <ctime>

#include "common/global_configs.h"

namespace surfelwarp {
	
	struct LogCheckError {
		LogCheckError() : str(nullptr) {}
		explicit LogCheckError(const std::string& str_p) : str(new std::string(str_p)) {}
		~LogCheckError() {
			if(str != nullptr) delete str;
		}
		//Type conversion
		operator bool() { return str != nullptr; }
		//The error string
		std::string* str;
	};

#define SURFELWARP_DEFINE_CHECK_FUNC(name, op)                      \
	template <typename X, typename Y>                               \
	inline LogCheckError LogCheck##name(const X& x, const Y& y) {   \
		if (x op y) return LogCheckError();                         \
        std::ostringstream os;                                      \
        os << " (" << x << " vs. " << y << ") ";                    \
        return LogCheckError(os.str());                             \
	}                                                               \
    inline LogCheckError LogCheck##name(int x, int y) {             \
        return LogCheck##name<int, int>(x, y);                      \
    }
SURFELWARP_DEFINE_CHECK_FUNC(_LT, <)
SURFELWARP_DEFINE_CHECK_FUNC(_GT, >)
SURFELWARP_DEFINE_CHECK_FUNC(_LE, <=)
SURFELWARP_DEFINE_CHECK_FUNC(_GE, >=)
SURFELWARP_DEFINE_CHECK_FUNC(_EQ, ==)
SURFELWARP_DEFINE_CHECK_FUNC(_NE, !=)



//Always on checking
#define SURFELWARP_CHECK(x)                                         \
	if(!(x))                                                        \
        surfelwarp::LogMessageFatal(__FILE__, __LINE__).stream()    \
        << "Check failed: " #x << " "

#define SURFELWARP_CHECK_BINARY_OP(name, op, x, y)                         \
	if(surfelwarp::LogCheckError err = surfelwarp::LogCheck##name(x, y))   \
        surfelwarp::LogMessageFatal(__FILE__, __LINE__).stream()           \
	    << "Check failed: " << #x " " #op " " #y << *(err.str)
	
#define	SURFELWARP_CHECK_LT(x, y) SURFELWARP_CHECK_BINARY_OP(_LT, < , x, y)
#define	SURFELWARP_CHECK_GT(x, y) SURFELWARP_CHECK_BINARY_OP(_GT, > , x, y)
#define	SURFELWARP_CHECK_LE(x, y) SURFELWARP_CHECK_BINARY_OP(_LE, <=, x, y)
#define	SURFELWARP_CHECK_GE(x, y) SURFELWARP_CHECK_BINARY_OP(_GE, >=, x, y)
#define	SURFELWARP_CHECK_EQ(x, y) SURFELWARP_CHECK_BINARY_OP(_EQ, ==, x, y)
#define	SURFELWARP_CHECK_NE(x, y) SURFELWARP_CHECK_BINARY_OP(_NE, !=, x, y)

//The log type for later use
#define LOG_INFO surfelwarp::LogMessage(__FILE__, __LINE__)
#define LOG_ERROR LOG_INFO
#define LOG_WARNING LOG_INFO
#define LOG_FATAL surfelwarp::LogMessageFatal(__FILE__, __LINE__)
#define LOG_BEFORE_THROW surfelwarp::LogMessage().stream()

//For different severity
#define LOG(severity) LOG_##severity.stream()
	
	// The log message
	class LogMessage {
	public:
		//Constructors
		LogMessage() : log_stream_(std::cout) {}
		LogMessage(const char* file, int line) : log_stream_(std::cout) {
			log_stream_ << file << ":" << line << ": ";
		}
		LogMessage(const LogMessage&) = delete;
		LogMessage& operator=(const LogMessage&) = delete;
		
		//Another line
		~LogMessage() { log_stream_ << "\n"; }
		
		std::ostream& stream() { return log_stream_; }
	protected:
		std::ostream& log_stream_;
	};
	
	class LogMessageFatal {
	public:
		LogMessageFatal(const char* file, int line) {
			log_stream_ << file << ":" << line << ": ";
		}
		
		//No copy/assign
		LogMessageFatal(const LogMessageFatal&) = delete;
		LogMessageFatal& operator=(LogMessageFatal&) = delete;
		
		//Die the whole system
		~LogMessageFatal() {
			LOG_BEFORE_THROW << log_stream_.str();
			throw new std::runtime_error(log_stream_.str());
		}
		
		//The output string stream
		std::ostringstream& stream() { return log_stream_; }
	protected:
		std::ostringstream log_stream_;
	};
	

} // namespace surfelwarp
