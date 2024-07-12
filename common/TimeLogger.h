#ifndef __TIME_LOG_H__
#define __TIME_LOG_H__

#include <chrono>
#include <string>
#include <mutex>

namespace surfelwarp {
class TimeLogger
{

public:
    using Tp = std::chrono::steady_clock::time_point;
    static void printTimeLog(
        const std::string& message, const std::string pre = "");

    static TimeLogger m_static_timer;
    static std::mutex m_mutex;

private:
    explicit TimeLogger();
    Tp t0, t1, t2;
    double duration0, duration1;

    void print(const std::string& message, const std::string pre = "");
};
}

// Macro definition
#ifdef PRINT_TIME_LOG
#define TIME_LOG(message) surfelwarp::TimeLogger::printTimeLog(message)
#else
#define TIME_LOG(message)
#endif

#endif // __TIME_LOG_H__
