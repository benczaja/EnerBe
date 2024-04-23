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

  /* VERY DUMB Argument Parsers */
  int N = parse_arguments(argc, argv, &simple, &openmp, &sanity_check);
  int ROWS = N;
  int COLUMNS = N;

  /* declare the arrays...  better to do it as 1D arrays for CUDA */

  // First allocated them on the host (CPU)
    X_TYPE* A = (X_TYPE*)malloc((ROWS * COLUMNS) * sizeof(X_TYPE));
    X_TYPE* B = (X_TYPE*)malloc((ROWS * COLUMNS) * sizeof(X_TYPE));
    X_TYPE* C = (X_TYPE*)malloc((ROWS * COLUMNS) * sizeof(X_TYPE));

  // Then Allocate them on the GPUs
  X_TYPE* D_A;
  X_TYPE* D_B;
  X_TYPE* D_C;
  cudaMalloc((void**)&D_A, sizeof( X_TYPE ) * (ROWS * COLUMNS));
  cudaMalloc((void**)&D_B, sizeof( X_TYPE ) * (ROWS * COLUMNS));
  cudaMalloc((void**)&D_C, sizeof( X_TYPE ) * (ROWS * COLUMNS));

  double start = omp_get_wtime();  

  initialize_matrix_1D(A, B, C, ROWS, COLUMNS);
    
  double end = omp_get_wtime(); 
  printf("Init TIME: %f sec\n",(end-start));

    // THIS IS NEW !!!!!!!
  auto GPUsensor = pmt::nvml::NVML::Create();
  auto CPUsensor = pmt::rapl::Rapl::Create();

  /*======================================================================*/
  /*                START of Section of the code that matters!!!          */
  /*======================================================================*/

  /* Simple matrix multiplication */
  /*==============================*/
    int block_size = 512;
    int grid_size = ((ROWS + block_size) / block_size);
    
    //Start the PMT "sensor"
    auto GPUstart = GPUsensor->Read(); // READING the GPU via NVML
    auto CPUstart = CPUsensor->Read(); // READING the CPU via RAPL

    // Transfer data from host to device memory
    cudaMemcpy(D_A, A, sizeof(X_TYPE) * (ROWS * COLUMNS), cudaMemcpyHostToDevice);
    cudaMemcpy(D_B, B, sizeof(X_TYPE) * (ROWS * COLUMNS), cudaMemcpyHostToDevice);
    
    simple_matrix_multiply<<<grid_size,block_size>>>(D_A, D_B, D_C, ROWS, COLUMNS);

   // Transfer data from device to host memory
    cudaMemcpy(C, D_C, sizeof(X_TYPE) * (ROWS * COLUMNS), cudaMemcpyDeviceToHost);


    //Start the PMT "sensor"
    auto GPUend = GPUsensor->Read();
    auto CPUend = CPUsensor->Read();

    std::cout << "SIZE: " << N << std::endl;
    std::cout << "(RAPL) CPU_TIME: " << pmt::PMT::seconds(CPUstart, CPUend) << " | (NVML) GPU_TIME: " << pmt::PMT::seconds(GPUstart, GPUend) << " s"<< std::endl;
    std::cout << "(RAPL) CPU_JOULES: " << pmt::PMT::joules(CPUstart, CPUend) << " | (NVML) GPU_JOULES: " << pmt::PMT::joules(GPUstart, GPUend) << " J"<< std::endl;
    std::cout << "(RAPL) CPU_WATTS: " << pmt::PMT::watts(CPUstart, CPUend) << " | (NVML) GPU_WATTS: " << pmt::PMT::watts(GPUstart, GPUend) << " W"<< std::endl;
    std::cout << "Total TIME: " << (pmt::PMT::seconds(CPUstart, CPUend) + pmt::PMT::seconds(GPUstart, GPUend))*0.5 << " s"<< std::endl;
    std::cout << "Total JOULES: " << (pmt::PMT::joules(CPUstart, CPUend) + pmt::PMT::joules(GPUstart, GPUend)) << " J"<< std::endl;
    std::cout << "Total WATTS: " << (pmt::PMT::watts(CPUstart, CPUend) + pmt::PMT::watts(GPUstart, GPUend)) << " W"<< std::endl;
    
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
