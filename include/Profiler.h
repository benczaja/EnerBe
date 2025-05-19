#include <ctime>
#include <omp.h>


class Profiler {
    public:
        double timestart = 0.0;
        double timeend = 0.0;

    double measureSerialTime(){
        // Sample the timer
        clock_t currentTime = clock();
        return (double) currentTime / CLOCKS_PER_SEC;
    }
    double measureOpenMPTime(){
        // Sample the timer
        return omp_get_wtime(); // Double
    }
};