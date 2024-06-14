#ifndef timer_hpp
#define timer_hpp

#include <chrono>
#include <iostream>
using namespace std::chrono;

class Timer
{
private:
    system_clock::time_point start;
    system_clock::time_point end;

public:
    Timer()
    {
        start = system_clock::now();
    }
    void reset()
    {
        start = system_clock::now();
    }
    void stop()
    {
        end = system_clock::now();
    }
    double get()
    {
        return duration_cast<nanoseconds>(end - start).count();
    }
    char print()
    {
        std::cout << get() * 1e-9 << " sec" << std::endl;
        return 0;
    }
};

#endif