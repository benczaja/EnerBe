#include <stdio.h> // needed for ‘printf’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
#include <pmt.h> // needed for PMT
#include <pmt/Rapl.h> // needed for RAPL
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include "kernals.h"

int main( int argc, char *argv[] )  {

    /* VERY DUMB Argument Parsers */
    int N = parse_arguments(argc, argv, &simple, &openmp, &sanity_check);

    X_TYPE *sx; /* n is an array of N integers */
    X_TYPE *sy; /* n is an array of N integers */

    sx = (X_TYPE*)malloc(N * sizeof (X_TYPE));
    sy = (X_TYPE*)malloc(N * sizeof (X_TYPE));

    // THIS IS NEW !!!!!!!
    auto sensor = pmt::rapl::Rapl::Create();
        
    /* Simple saxpy */
    /*==============================*/
    if (true == simple)
    {

        //Start the PMT "sensor"
        auto start = sensor->Read();
    
        simple_axpy(N, 2.0, sx, sy);

        //End the PMT "sensor"
        auto end = sensor->Read();

        std::cout << "CLASS: axpy" << std::endl;
        std::cout << "ALGO: simple" << std::endl;
        std::cout << "PRECISION: "<< sizeof (X_TYPE) <<" bytes" << std::endl;
        std::cout << "SIZE: " << N <<std::endl;
        std::cout << "(RAPL) CPU_TIME: " << pmt::PMT::seconds(start, end) << " s"<< std::endl;
        std::cout << "(RAPL) CPU_JOULES: " << pmt::PMT::joules(start, end) << " J" << std::endl;
        std::cout << "(RAPL) CPU_WATTS: " << pmt::PMT::watts(start, end) << " W" << std::endl;

    }

    /* OpenMP parallel saxpy */
    /*==============================*/
    if (true == openmp)
    {

        //Start the PMT "sensor"
        auto start = sensor->Read();

        openmp_axpy(N, 2.0, sx, sy);
    
        //End the PMT "sensor"
        auto end = sensor->Read();

        /// SORRY FOR THE CPP !!!!! BUT WE ARE JUST PRINTING!!!!
        std::cout << "CLASS: axpy" << std::endl;
        std::cout << "ALGO: openmp" << std::endl;
        std::cout << "PRECISION: "<< sizeof (X_TYPE) <<" bytes" << std::endl;
        std::cout << "OMP_THREADS: "<< omp_get_max_threads() << std::endl;
        std::cout << "SIZE: " << N <<std::endl;
        std::cout << "(RAPL) CPU_TIME: " << pmt::PMT::seconds(start, end) << " s"<< std::endl;
        std::cout << "(RAPL) CPU_JOULES: " << pmt::PMT::joules(start, end) << " J" << std::endl;
        std::cout << "(RAPL) CPU_WATTS: " << pmt::PMT::watts(start, end) << " W" << std::endl;

    }


}
