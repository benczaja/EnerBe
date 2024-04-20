#include "hip/hip_runtime.h"
#include <stdio.h> // needed for ‘printf’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code

#ifdef USE_DOUBLE
typedef double X_TYPE;
#else
typedef float X_TYPE;
#endif

void simple_axpy(int n, X_TYPE a, X_TYPE * x, X_TYPE * y){

    printf("(Simple) saxpy of Array of size (%d)\n",n);

    for(int i=0; i<n; i++){
        y[i] = a * x[i] + y[i];
    }
}

void openmp_axpy(int n, X_TYPE a, X_TYPE * x, X_TYPE * y){
    int num_threads = omp_get_max_threads();

    printf("(OpenMP) saxpy of Array of size (%d)\n",n);
    printf("Using %d Threads\n", num_threads);
    #pragma omp parallel for
    for(int i=0; i<n; i++){
        y[i] = a * x[i] + y[i];
    }
}

__global__ void gpu_axpy(int n, X_TYPE a, X_TYPE * x, X_TYPE * y) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handling arbitrary vector size
    if (tid < n){
        y[tid] = a * x[tid] + y[tid];
    }
}





int main( int argc, char *argv[] )  {

    printf("X_TYPE size is (%d) bytes \n",sizeof (X_TYPE));

    int N;
    clock_t t; // declare clock_t (long type)
    /* DUMB bools needed for the argument parsing logic */
    bool openmp = false;
    bool simple = true;
    bool sanity_check = false;
    
    /* VERY DUMB Argument Parsers */
    N = parse_arguments(argc, argv, &simple, &openmp, &sanity_check);

    X_TYPE *d_sx; /* n is an array of N integers */
    X_TYPE *d_sy; /* n is an array of N integers */

    X_TYPE a = 2.0;
    // Allocate Host memory 
    X_TYPE* sx = (X_TYPE*)malloc(N * sizeof(X_TYPE));
    X_TYPE* sy = (X_TYPE*)malloc(N * sizeof(X_TYPE));

    // Allocate device memory 
    hipMalloc((void**)&d_sx, sizeof(X_TYPE) * N);
    hipMalloc((void**)&d_sy, sizeof(X_TYPE) * N);


    /* Simple saxpy */
    /*==============================*/
    if (true == simple)
    {
    int block_size = 512;
    int grid_size = ((ROWS + block_size) / block_size);
    
    t = clock(); // start the clock

    // Transfer data from host to device memory
    hipMemcpy(D_A, A, sizeof(X_TYPE) * (ROWS * COLUMNS), hipMemcpyHostToDevice);
    hipMemcpy(D_B, B, sizeof(X_TYPE) * (ROWS * COLUMNS), hipMemcpyHostToDevice);
    
    gpu_axpy<<<grid_size,block_size>>>(N, a, d_sx, d_sy);

    // Transfer data from device to host memory
    hipMemcpy(C, D_C, sizeof(X_TYPE) * (ROWS * COLUMNS), hipMemcpyDeviceToHost);

    t = clock() - t; // stop the clock

    double time_taken = ((double)t)/CLOCKS_PER_SEC; // convert to seconds (and long to double)
    printf("GPU Compute Time: %f s\n",time_taken);
  }

    }
    /* OpenMP parallel saxpy */
    /*==============================*/
    if (true == openmp)
    {

    // omp_get_wtime needed here because clock will sum up time for all threads
    double start = omp_get_wtime();  

    openmp_axpy(N, 2.0, sx, sy);
    
    double end = omp_get_wtime(); 
    printf("TIME: %f sec\n",(end-start));

    }


}
