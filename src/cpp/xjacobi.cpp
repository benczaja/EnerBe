#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘RAND_MAX’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
#include "kernals.h"


int main( int argc, char *argv[] )  {
    
  kernal kernal;
  kernal.name = "xjacobi";
  
  /* VERY DUMB Argument Parsers */
  kernal.size = parse_arguments(argc, argv);

  /* declare the arrays */
  X_TYPE* A = (X_TYPE*)malloc(kernal.size * kernal.size * sizeof(X_TYPE));
  X_TYPE* B = (X_TYPE*)malloc((kernal.size) * sizeof(X_TYPE));
  X_TYPE* C = (X_TYPE*)malloc((kernal.size) * sizeof(X_TYPE));

  /*======================================================================*/
  /*                START of Section of the code that matters!!!          */
  /*======================================================================*/

  /* initialize the arrays */
  //initialize_matrix_1D(A, B, C, kernal.size, kernal.size);
  srand(235);
  for (int row = 0; row < kernal.size; row++)
  {
    double rowsum = 0.0;
    for (int col = 0; col < kernal.size; col++)
    {
      double value = rand()/(double)RAND_MAX;
      A[row + col*kernal.size] = value;
      rowsum += value;
    }
    A[row + row*kernal.size] += rowsum;
    B[row] = rand()/(double)RAND_MAX;
    C[row] = 0.0;
  }

  /* Simple matrix multiplication */
  /*==============================*/
  if (true == simple)
  {
    clock_t t; // declare clock_t (long type)
    kernal.algorithm = "simple";
    kernal.omp_threads = 1;
    do {
      kernal.start = double(clock());
      simple_jacobi(A, B, C, kernal.size, kernal.size);
      kernal.end = double(clock());
      kernal.times[kernal.N_runs] =  (kernal.end - kernal.start)/CLOCKS_PER_SEC;
      kernal.N_runs ++;
    }while (kernal.time < kernal.max_time && kernal.N_runs < kernal.max_runs);


  // Check error of final solution
  double err = 0.0;
  for (int row = 0; row < kernal.size; row++)
  {
    double tmp = 0.0;
    for (int col = 0; col < kernal.size; col++)
    {
      tmp += A[row + col*kernal.size] * C[col];
    }
    tmp = B[row] - tmp;
    err += tmp*tmp;
  }
    err = sqrt(err);
    std::cout<< "error: "<< err << std::endl;

    kernal.calculate_stats();
    kernal.print_info();
  }
  /* deallocate the arrays */
  free(A);
  free(B);
  free(C);
}
