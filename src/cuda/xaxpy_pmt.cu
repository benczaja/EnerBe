#include <stdio.h> // needed for ‘printf’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
#include <pmt.h> // needed for PMT
#include <pmt/Rapl.h> // needed for RAPL
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include "kernals.h"

int main( int argc, char *argv[] )  {

    kernal kernal;
    kernal.name = "axpy";

    /* VERY DUMB Argument Parsers */
    kernal.size = parse_arguments(argc, argv, &simple, &openmp, &sanity_check);

    X_TYPE *d_sx; /* n is an array of N integers */
    X_TYPE *d_sy; /* n is an array of N integers */

    X_TYPE a = 2.0;
    // Allocate Host memory 
    X_TYPE* sx = (X_TYPE*)malloc(kernal.size * sizeof(X_TYPE));
    X_TYPE* sy = (X_TYPE*)malloc(kernal.size * sizeof(X_TYPE));

    // Allocate device memory 
    cudaMalloc((void**)&d_sx, sizeof(X_TYPE) * kernal.size);
    cudaMalloc((void**)&d_sy, sizeof(X_TYPE) * kernal.size);

    // THIS IS NEW !!!!!!!
    auto GPUsensor = pmt::nvml::NVML::Create();
    auto CPUsensor = pmt::rapl::Rapl::Create();

    /* Simple saxpy */
    /*==============================*/
    kernal.algorithm = "simple_gpu";
    cudaGetDevice(&kernal.gpuid);  

    int block_size = 512;
    int grid_size = ((kernal.size + block_size) / block_size);
    
    //Start the PMT "sensor"
    auto GPUstart = GPUsensor->Read(); // READING the GPU via NVML
    auto CPUstart = CPUsensor->Read(); // READING the CPU via RAPL
    
    // Transfer data from host to device memory
    cudaMemcpy(d_sx, sx, sizeof(X_TYPE) * kernal.size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sy, sy, sizeof(X_TYPE) * kernal.size, cudaMemcpyHostToDevice);
    
    gpu_axpy<<<grid_size,block_size>>>(kernal.size, a, d_sx, d_sy);
    
    cudaMemcpy(sy, d_sy, sizeof(X_TYPE) * kernal.size, cudaMemcpyDeviceToHost);
    
    //Start the PMT "sensor"
    auto GPUend = GPUsensor->Read();
    auto CPUend = CPUsensor->Read();

    kernal.rapl_time = pmt::PMT::seconds(CPUstart, CPUend);
    kernal.rapl_power = pmt::PMT::watts(CPUstart, CPUend);
    kernal.rapl_energy = pmt::PMT::joules(CPUstart, CPUend);

    kernal.nvml_time = pmt::PMT::seconds(GPUstart, GPUend);
    kernal.nvml_power = pmt::PMT::watts(GPUstart, GPUend);
    kernal.nvml_energy = pmt::PMT::joules(GPUstart, GPUend);

    kernal.print_pmt_nvml_info();
}
