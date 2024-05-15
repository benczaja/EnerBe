#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘RAND_MAX’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
#include <pmt.h> // needed for PMT
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include "kernals.h"

int main( int argc, char *argv[] )  {

  kernal kernal;
  kernal.name = "xgemm";

  /* VERY DUMB Argument Parsers */
  kernal.size = parse_arguments(argc, argv);

  /* declare the arrays */
  X_TYPE** A = (X_TYPE**)malloc(kernal.size * sizeof( X_TYPE* ));
  X_TYPE** B = (X_TYPE**)malloc(kernal.size * sizeof( X_TYPE* ));
  X_TYPE** C = (X_TYPE**)malloc(kernal.size * sizeof( X_TYPE* ));

    for (int i =0; i <kernal.size; i++)
    {
        A[i] = (X_TYPE*)malloc(kernal.size * sizeof(X_TYPE));
        B[i] = (X_TYPE*)malloc(kernal.size * sizeof(X_TYPE));
        C[i] = (X_TYPE*)malloc(kernal.size * sizeof(X_TYPE));
    }

  /*======================================================================*/
  /*                START of Section of the code that matters!!!          */
  /*======================================================================*/

  /* initialize the arrays */
  initialize_matrix_2D(A, B, C, kernal.size, kernal.size);

  // THIS IS NEW !!!!!!!
  //std::unique_ptr<pmt::PMT> sensor = pmt::rapl::Rapl::Create();
  std::unique_ptr<pmt::PMT> sensor = pmt::Create("Rapl");
  auto start = sensor->Read();
  auto end = sensor->Read();

  /* Simple matrix multiplication */
  /*==============================*/
  if (true == simple)
  {
    kernal.algorithm = "simple";
    kernal.omp_threads = 1;
    //Read from the PMT "sensor"
    do {
      start = sensor->Read();
      simple_matrix_multiply(A, B, C, kernal.size, kernal.size);
      //Read from the PMT "sensor"
      end = sensor->Read();
      kernal.rapl_times[kernal.N_runs] = pmt::PMT::seconds(start, end);
      kernal.rapl_powers[kernal.N_runs] = pmt::PMT::watts(start, end);
      kernal.rapl_energys[kernal.N_runs] = pmt::PMT::joules(start, end);
      kernal.N_runs ++;
    }while (kernal.time < kernal.max_time && kernal.N_runs < kernal.max_runs);
    kernal.calculate_stats();
  }

  /* OpenMP parallel matrix multiplication */
  /*=======================================*/
  if (true == openmp)
  {
    kernal.algorithm = "openmp";
    do{
      //Read from the PMT "sensor"
      start = sensor->Read();
      openmp_matrix_multiply(A, B, C, kernal.size, kernal.size);
      //Read from the PMT "sensor"
      end = sensor->Read();
      kernal.rapl_times[kernal.N_runs] = pmt::PMT::seconds(start, end);
      kernal.rapl_powers[kernal.N_runs] = pmt::PMT::watts(start, end);
      kernal.rapl_energys[kernal.N_runs] = pmt::PMT::joules(start, end);
      kernal.N_runs ++;
    }while (kernal.time < kernal.max_time && kernal.N_runs < kernal.max_runs);
    kernal.calculate_stats();
  }
    kernal.print_pmt_rapl_info();

  /*======================================================================*/
  /*                 END of Section of the code that matters!!!           */
  /*======================================================================*/

  /* deallocate the arrays */
  for (int i=0; i<kernal.size; i++)
  {
    free(A[i]);
    free(B[i]);
    free(C[i]);
  }
  free(A);
  free(B);
  free(C);
}
