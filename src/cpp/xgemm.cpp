#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘RAND_MAX’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
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

  /* Simple matrix multiplication */
  /*==============================*/
  if (true == simple)
  {
    kernal.algorithm = "simple";
    kernal.omp_threads = 1;
    clock_t t; // declare clock_t (long type)
    t = clock(); // start the clock
    
    simple_matrix_multiply(A, B, C, kernal.size, kernal.size);
    
    t = clock() - t; // stop the clock

    kernal.time = ((double)t)/CLOCKS_PER_SEC; // convert to seconds (and long to double)
  }

  /* OpenMP parallel matrix multiplication */
  /*=======================================*/
  if (true == openmp)
  {
    kernal.algorithm = "openmp";

    // omp_get_wtime needed here because clock will sum up time for all threads
    double start = omp_get_wtime();  

    openmp_matrix_multiply(A, B, C, kernal.size, kernal.size);
    
    double end = omp_get_wtime(); 
    kernal.time = (end-start);
  
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
