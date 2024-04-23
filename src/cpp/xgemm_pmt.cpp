#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘RAND_MAX’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
#include <pmt.h> // needed for PMT
#include <pmt/Rapl.h> // needed for RAPL
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include "kernals.h"

int main( int argc, char *argv[] )  {

  double time_taken=0;
  
  /* VERY DUMB Argument Parsers */
  int N = parse_arguments(argc, argv, &simple, &openmp, &sanity_check);
  int ROWS = N;
  int COLUMNS = N;

  /* declare the arrays */
  X_TYPE** A = (X_TYPE**)malloc(ROWS * sizeof( X_TYPE* ));
  X_TYPE** B = (X_TYPE**)malloc(ROWS * sizeof( X_TYPE* ));
  X_TYPE** C = (X_TYPE**)malloc(ROWS * sizeof( X_TYPE* ));

    for (int i =0; i <ROWS; i++)
    {
        A[i] = (X_TYPE*)malloc(COLUMNS * sizeof(X_TYPE));
        B[i] = (X_TYPE*)malloc(COLUMNS * sizeof(X_TYPE));
        C[i] = (X_TYPE*)malloc(COLUMNS * sizeof(X_TYPE));
    }

  /*======================================================================*/
  /*                START of Section of the code that matters!!!          */
  /*======================================================================*/

  /* initialize the arrays */
  initialize_matrix_2D(A, B, C, ROWS, COLUMNS);

  // THIS IS NEW !!!!!!!
  auto sensor = pmt::rapl::Rapl::Create();

  /* Simple matrix multiplication */
  /*==============================*/
  if (true == simple)
  {
    //Start the PMT "sensor"
    auto start = sensor->Read();

    simple_matrix_multiply(A, B, C, ROWS, COLUMNS);
    
    //End the PMT "sensor"
    auto end = sensor->Read();

    /// SORRY FOR THE CPP !!!!! BUT WE ARE JUST PRINTING!!!!
    std::cout << "ALGO: simple"<< std::endl;
    std::cout << "PRECISION: "<< sizeof (X_TYPE) <<" bytes"<< std::endl;
    std::cout << "SIZE: " << N <<std::endl;
    std::cout << "(RAPL) CPU_TIME: " << pmt::PMT::seconds(start, end) << " s"<< std::endl;
    std::cout << "(RAPL) CPU_JOULES: " << pmt::PMT::joules(start, end) << " J" << std::endl;
    std::cout << "(RAPL) CPU_WATTS: " << pmt::PMT::watts(start, end) << " W" << std::endl;


  }

  /* OpenMP parallel matrix multiplication */
  /*=======================================*/
  if (true == openmp)
  {
    //Start the PMT "sensor"
    auto start = sensor->Read();

    openmp_matrix_multiply(A, B, C, ROWS, COLUMNS);
    
    //End the PMT "sensor"
    auto end = sensor->Read();

    /// SORRY FOR THE CPP !!!!! BUT WE ARE JUST PRINTING!!!!
    std::cout << "ALGO: openmp"<< std::endl;
    std::cout << "PRECISION: "<< sizeof (X_TYPE) <<" bytes"<< std::endl;
    std::cout << "OMP_THREADS: "<< omp_get_max_threads() << std::endl;
    std::cout << "SIZE: " << N <<std::endl;
    std::cout << "(RAPL) CPU_TIME: " << pmt::PMT::seconds(start, end) << " s"<< std::endl;
    std::cout << "(RAPL) CPU_JOULES: " << pmt::PMT::joules(start, end) << " J" << std::endl;
    std::cout << "(RAPL) CPU_WATTS: " << pmt::PMT::watts(start, end) << " W" << std::endl;

  }

  /*======================================================================*/
  /*                 END of Section of the code that matters!!!           */
  /*======================================================================*/

  /* deallocate the arrays */
  for (int i=0; i<ROWS; i++)
  {
    free(A[i]);
    free(B[i]);
    free(C[i]);
  }
  free(A);
  free(B);
  free(C);
}
