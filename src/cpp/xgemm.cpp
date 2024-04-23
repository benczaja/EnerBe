#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘RAND_MAX’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
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

  /* Simple matrix multiplication */
  /*==============================*/
  if (true == simple)
  {
    clock_t t; // declare clock_t (long type)
    t = clock(); // start the clock
    
    simple_matrix_multiply(A, B, C, ROWS, COLUMNS);
    
    t = clock() - t; // stop the clock

    time_taken = ((double)t)/CLOCKS_PER_SEC; // convert to seconds (and long to double)
    
    printf("ALGO: simple\n");
    printf("PRECISION: %d bytes \n",sizeof (X_TYPE));
    printf("SIZE: %d \n",ROWS);
    printf("TIME: %f s\n",time_taken);
  }

  /* OpenMP parallel matrix multiplication */
  /*=======================================*/
  if (true == openmp)
  {
    // omp_get_wtime needed here because clock will sum up time for all threads
    double start = omp_get_wtime();  

    openmp_matrix_multiply(A, B, C, ROWS, COLUMNS);
    
    double end = omp_get_wtime(); 
    time_taken = (end-start);

    printf("ALGO: openmp\n");
    printf("PRECISION: %d bytes \n",sizeof (X_TYPE));
    printf("OMP_THREADS: %d\n",omp_get_max_threads());
    printf("SIZE: %d \n",ROWS);
    printf("TIME: %f s\n",time_taken);    
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
