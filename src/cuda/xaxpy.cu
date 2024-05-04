#include <stdio.h> // needed for ‘printf’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
#include "kernals.h"


int main( int argc, char *argv[] )  {
    
    clock_t t; // declare clock_t (long type)
    kernal kernal;
    kernal.name = "axpy";

    /* VERY DUMB Argument Parsers */
    kernal.size = parse_arguments(argc, argv);

    X_TYPE *d_sx; /* n is an array of N integers */
    X_TYPE *d_sy; /* n is an array of N integers */

    X_TYPE a = 2.0;
    // Allocate Host memory 
    X_TYPE* sx = (X_TYPE*)malloc(kernal.size * sizeof(X_TYPE));
    X_TYPE* sy = (X_TYPE*)malloc(kernal.size * sizeof(X_TYPE));


    // Allocate device memory 
    cudaMalloc((void**)&d_sx, sizeof(X_TYPE) * kernal.size);
    cudaMalloc((void**)&d_sy, sizeof(X_TYPE) * kernal.size);
        
        
    kernal.algorithm = "simple_gpu";
    cudaGetDevice(&kernal.gpuid);  
    int block_size = 512;
    int grid_size = ((kernal.size + block_size) / block_size);

if (true == simple)
  {
    do{
    kernal.start = double(clock()); // start the clock

    // Transfer data from host to device memory
    cudaMemcpy(d_sx, sx, sizeof(X_TYPE) * kernal.size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sy, sy, sizeof(X_TYPE) * kernal.size, cudaMemcpyHostToDevice);

    gpu_axpy<<<grid_size,block_size>>>(kernal.size, a, d_sx, d_sy);

    cudaMemcpy(sy, d_sy, sizeof(X_TYPE) * kernal.size, cudaMemcpyDeviceToHost);
    
    kernal.end = double(clock()); // stop the clock
      kernal.times[kernal.N_runs] =  (kernal.end - kernal.start)/CLOCKS_PER_SEC;
      kernal.N_runs ++;
    }while (kernal.time < kernal.max_time && kernal.N_runs < kernal.max_runs);
    kernal.calculate_stats();
  }
  kernal.print_info();

    cudaFree(d_sx);
    cudaFree(d_sy);

    free(sx);
    free(sy);
}
