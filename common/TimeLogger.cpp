#include "common/TimeLogger.h"
#include <iomanip>
#include <iostream>

namespace surfelwarp{

    // constructor
    TimeLogger::TimeLogger(){
        t0 = t1 = t2 = std::chrono::steady_clock::now();
        duration0 = duration1 = 0.0; 
    }

    // private print
    void TimeLogger::print(const std::string & message, const std::string pre){
        t2 = std::chrono::steady_clock::now();
        duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();
        duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << std::setw(10) << duration1 << "   "  << std::setw(15) << duration0 / 1e6 << "   " << pre << message << std::endl;
        t1 = t2;
    }


    // static public print
    void TimeLogger::printTimeLog(const std::string& message, const std::string pre){
        // xt: does it work to add unique_lock?
        std::unique_lock<std::mutex> lock(TimeLogger::m_mutex);
        m_static_timer.print(message, pre);
    }

    // static data
    TimeLogger TimeLogger::m_static_timer = TimeLogger();

    std::mutex TimeLogger::m_mutex;

}
