#include <stdio.h> // needed for ‘printf’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
#include <pmt.h> // needed for PMT
#include <pmt/Rapl.h> // needed for RAPL
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include "kernals.h"

int main( int argc, char *argv[] )  {

    kernal kernal;
    kernal.name = "axpy";

    /* VERY DUMB Argument Parsers */
    kernal.size = parse_arguments(argc, argv);
    X_TYPE *sx; /* n is an array of N integers */
    X_TYPE *sy; /* n is an array of N integers */

    sx = (X_TYPE*)malloc(kernal.size * sizeof (X_TYPE));
    sy = (X_TYPE*)malloc(kernal.size * sizeof (X_TYPE));

    // THIS IS NEW !!!!!!!
    auto sensor = pmt::rapl::Rapl::Create();
    auto start = sensor->Read();
    auto end = sensor->Read();
    
    /* Simple saxpy */
    /*==============================*/
    if (true == simple)
    {
        kernal.algorithm = "simple";
        //Read from the PMT "sensor"
        start = sensor->Read();
    
        simple_axpy(kernal.size, 2.0, sx, sy);

        //Read from the PMT "sensor"
        end = sensor->Read();

    }

    /* OpenMP parallel saxpy */
    /*==============================*/
    if (true == openmp)
    {
        kernal.algorithm = "openmp";
        //Read from the PMT "sensor"
        auto start = sensor->Read();

        openmp_axpy(kernal.size, 2.0, sx, sy);
    
        //Read from the PMT "sensor"
        end = sensor->Read();
    }

    kernal.rapl_time = pmt::PMT::seconds(start, end);
    kernal.rapl_power = pmt::PMT::watts(start, end);
    kernal.rapl_energy = pmt::PMT::joules(start, end);

    kernal.print_pmt_rapl_info();   

    free(sx);
    free(sy);
}
