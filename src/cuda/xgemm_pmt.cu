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
  printf("Init TIME: %f sec\n",(end-start));

  // THIS IS NEW !!!!!!!
  std::unique_ptr<pmt::PMT> GPUsensor = pmt::nvml::NVML::Create();
  std::unique_ptr<pmt::PMT> CPUsensor = pmt::rapl::Rapl::Create();
  //Start the PMT "sensor"
  auto GPUstart = GPUsensor->Read(); 
  auto CPUstart = CPUsensor->Read(); 
  auto GPUend = GPUsensor->Read(); 
  auto CPUend = CPUsensor->Read(); 

  /*======================================================================*/
  /*                START of Section of the code that matters!!!          */
  /*======================================================================*/

  /* Simple matrix multiplication */
  /*==============================*/
    if (true ==simple){
    kernal.algorithm = "simple_gpu";
    cudaGetDevice(&kernal.gpuid);  	
  
    int block_size = 512;
    int grid_size = ((kernal.size + block_size) / block_size);
    
    do {
    //Start the PMT "sensor"
    GPUstart = GPUsensor->Read(); // READING the GPU via NVML
    CPUstart = CPUsensor->Read(); // READING the CPU via RAPL

    // Transfer data from host to device memory
    cudaMemcpy(D_A, A, sizeof(X_TYPE) * (kernal.size * kernal.size), cudaMemcpyHostToDevice);
    cudaMemcpy(D_B, B, sizeof(X_TYPE) * (kernal.size * kernal.size), cudaMemcpyHostToDevice);
    
    simple_matrix_multiply<<<grid_size,block_size>>>(D_A, D_B, D_C, kernal.size, kernal.size);

   // Transfer data from device to host memory
    cudaMemcpy(C, D_C, sizeof(X_TYPE) * (kernal.size * kernal.size), cudaMemcpyDeviceToHost);

    //END the PMT "sensor"
    GPUend = GPUsensor->Read();
    CPUend = CPUsensor->Read();

    kernal.rapl_times[kernal.N_runs] = pmt::PMT::seconds(CPUstart, CPUend);
    kernal.rapl_powers[kernal.N_runs] = pmt::PMT::watts(CPUstart, CPUend);
    kernal.rapl_energys[kernal.N_runs] = pmt::PMT::joules(CPUstart, CPUend);

    kernal.nvml_times[kernal.N_runs] = pmt::PMT::seconds(GPUstart, GPUend);
    kernal.nvml_powers[kernal.N_runs] = pmt::PMT::watts(GPUstart, GPUend);
    kernal.nvml_energys[kernal.N_runs] = pmt::PMT::joules(GPUstart, GPUend);

    kernal.N_runs ++;
    }while (kernal.time < kernal.max_time && kernal.N_runs < kernal.max_runs);
    kernal.calculate_stats();
  }
    kernal.print_pmt_nvml_info();


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
