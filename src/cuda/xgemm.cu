#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘RAND_MAX’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
#include "kernals.h"

int main( int argc, char *argv[] )  {

  kernal kernal;
  kernal.name = "xgemm";
  clock_t t; // declare clock_t (long type)

  /* VERY DUMB Argument Parsers */
  kernal.size = parse_arguments(argc, argv, &simple, &openmp, &sanity_check);

  /* declare the arrays...  better to do it as 1D arrays for CUDA */
  // First allocated them on the host (CPU)
  X_TYPE* A = (X_TYPE*)malloc((kernal.size * kernal.size) * sizeof(X_TYPE));
  X_TYPE* B = (X_TYPE*)malloc((kernal.size * kernal.size) * sizeof(X_TYPE));
  X_TYPE* C = (X_TYPE*)malloc((kernal.size * kernal.size) * sizeof(X_TYPE));

  // Then Allocate them on the GPUs
  X_TYPE* D_A;
  X_TYPE* D_B;
  X_TYPE* D_C;
  cudaMalloc((void**)&D_A, sizeof( X_TYPE ) * (kernal.size * kernal.size));
  cudaMalloc((void**)&D_B, sizeof( X_TYPE ) * (kernal.size * kernal.size));
  cudaMalloc((void**)&D_C, sizeof( X_TYPE ) * (kernal.size * kernal.size));

  double start = omp_get_wtime();  

  initialize_matrix_1D(A, B, C, kernal.size, kernal.size);
    
  double end = omp_get_wtime(); 
  printf("Init TIME: %f s\n",(end-start));


  /*======================================================================*/
  /*                START of Section of the code that matters!!!          */
  /*======================================================================*/

  /* Simple matrix multiplication */
  /*==============================*/
    kernal.algorithm = "simple_gpu";
    cudaGetDevice(&kernal.gpuid);  	

    int block_size = 512;
    int grid_size = ((kernal.size + block_size) / block_size);
    
    t = clock(); // start the clock

    // Transfer data from host to device memory
    cudaMemcpy(D_A, A, sizeof(X_TYPE) * (kernal.size * kernal.size), cudaMemcpyHostToDevice);
    cudaMemcpy(D_B, B, sizeof(X_TYPE) * (kernal.size * kernal.size), cudaMemcpyHostToDevice);
    
    simple_matrix_multiply<<<grid_size,block_size>>>(D_A, D_B, D_C, kernal.size, kernal.size);

   // Transfer data from device to host memory
    cudaMemcpy(C, D_C, sizeof(X_TYPE) * (kernal.size * kernal.size), cudaMemcpyDeviceToHost);

    t = clock() - t; // stop the clock

    kernal.time = ((double)t)/CLOCKS_PER_SEC; // convert to seconds (and long to double)

    kernal.print_info();
  /*======================================================================*/
  /*                 END of Section of the code that matters!!!           */
  /*======================================================================*/

 // Deallocate device memory
    cudaFree(D_A);
    cudaFree(D_B);
    cudaFree(D_C);

  // Deallocate host memory
  free(A);
  free(B);
  free(C);
}
