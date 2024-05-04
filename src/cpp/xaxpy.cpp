#include <stdio.h> // needed for ‘printf’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
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

    /* Simple saxpy */
    /*==============================*/
  if (true == simple)
  {
    clock_t t; // declare clock_t (long type)
    kernal.algorithm = "simple";
    kernal.omp_threads = 1;
    do {
      
      kernal.start = double(clock());
      simple_axpy(kernal.size, 2.0, sx, sy);
      kernal.end = double(clock());

      kernal.times[kernal.N_runs] =  (kernal.end - kernal.start)/CLOCKS_PER_SEC;
      kernal.N_runs ++;

    }while (kernal.time < kernal.max_time && kernal.N_runs < kernal.max_runs);
    kernal.calculate_stats();
  }
    /* OpenMP parallel saxpy */
    /*==============================*/
  if (true == openmp)
  {
    kernal.algorithm = "openmp";
    // omp_get_wtime needed here because clock will sum up time for all threads
    do {
      kernal.start = omp_get_wtime();  
      openmp_axpy(kernal.size, 2.0, sx, sy);
      kernal.end = omp_get_wtime(); 
      kernal.times[kernal.N_runs] += kernal.end - kernal.start;
      kernal.N_runs ++;
    }while (kernal.time < kernal.max_time && kernal.N_runs < kernal.max_runs);
    kernal.calculate_stats();
  }
  kernal.print_info();
    free(sx);
    free(sy);
}
