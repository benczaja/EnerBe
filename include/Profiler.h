#include <ctime>


#include <time.h>

class Profiler {
    public:
        double timestart = 0.0;
        double timeend = 0.0;

    double measureTime(){
        // Start the timer
        clock_t currentTime = clock();
        return (double) currentTime / CLOCKS_PER_SEC;
    }
    double calculateTime(){
        // Start the timer
        clock_t currentTime = clock();
        return (double) currentTime / CLOCKS_PER_SEC;
    }
};