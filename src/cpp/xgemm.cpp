#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘RAND_MAX’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
//#include "../EnerBe.h" // local helper header to clean up code
#include "../mesh.h" // local helper header to clean up code
#include "../argparser.h"
#include "kernals.h"


int main( int argc, char *argv[] )  {
    
  mesh2d kernal;
  parse_arguments(argc, argv, kernal.size, kernal.algorithm, kernal.name);

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
  kernal.initialize_matrix_2D(A, B, C, kernal.size, kernal.size);

  /* Simple matrix multiplication */
  /*==============================*/
  if (kernal.name == "xgemm" && kernal.algorithm == "simple")
  {
    clock_t t; // declare clock_t (long type)
    kernal.omp_threads = 1;
    do {
      kernal.start = double(clock());
      simple_matrix_multiply(A, B, C, kernal.size, kernal.size);
      kernal.end = double(clock());
      kernal.times[kernal.N_runs] =  (kernal.end - kernal.start)/CLOCKS_PER_SEC;
      kernal.N_runs ++;
    }while (kernal.time < kernal.max_time && kernal.N_runs < kernal.max_runs);
    kernal.calculate_stats();
  }
  /* OpenMP parallel matrix multiplication */
  /*=======================================*/
  if (kernal.name == "xgemm" && kernal.algorithm == "openmp")
  {
    kernal.algorithm = "openmp";
    // omp_get_wtime needed here because clock will sum up time for all threads
    do {
      kernal.start = omp_get_wtime();  
      openmp_matrix_multiply(A, B, C, kernal.size, kernal.size);
      kernal.end = omp_get_wtime(); 
      kernal.times[kernal.N_runs] += kernal.end - kernal.start;
      kernal.N_runs ++;
    }while (kernal.time < kernal.max_time && kernal.N_runs < kernal.max_runs);
    kernal.calculate_stats();
  }
  kernal.print_info();

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
