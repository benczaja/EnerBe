#include <ctime>
#include <omp.h>
#include <iostream>
#include <memory>
#include <pmt.h> // needed for PMT


class Profiler {
    public:
        bool measured = false;

        double time = 0.0;
        double power = 0.0;
        double energy = 0.0;

        double timestart = 0.0;
        double timeend = 0.0;

        // PMT stuff
        std::unique_ptr<pmt::PMT> RAPLsensor = pmt::rapl::Rapl::Create();
        pmt::State RAPLstart = RAPLsensor->Read();
        pmt::State RAPLend = RAPLsensor->Read();

    void measureSerialTime(){

        if (!measured) {
            clock_t currentTime = clock();
            timestart = (double) currentTime / CLOCKS_PER_SEC;
            measured = true;
        }else{
            clock_t currentTime = clock();
            timeend = (double) currentTime / CLOCKS_PER_SEC;
            time = timeend - timestart;
            measured = false;
        }
    }
    void measureOpenMPTime(){

        if (!measured) {
            timestart = omp_get_wtime();
            measured = true;
        }else{
            timeend = omp_get_wtime();
            time = timeend - timestart;
            measured = false;
        }
    }

    void measureRAPL(){
        if (!measured) {
            RAPLstart = RAPLsensor->Read();
            measured = true;
        }else{
            RAPLend = RAPLsensor->Read();
            time = pmt::PMT::seconds(RAPLstart, RAPLend);
            power = pmt::PMT::watts(RAPLstart, RAPLend);
            energy = pmt::PMT::joules(RAPLstart, RAPLend);
            measured = false;
        }
    }

    void printResults() const {
        std::cout << "[Profiler] Time: " << time << " seconds\n";
        std::cout << "[Profiler] Power: " << power << " W\n";
        std::cout << "[Profiler] Energy: " << energy << " J\n";
    }
};