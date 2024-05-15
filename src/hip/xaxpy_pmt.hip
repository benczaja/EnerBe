#include "hip/hip_runtime.h"
#include <stdio.h> // needed for ‘printf’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
#include <pmt.h> // needed for PMT
#include <pmt/Rapl.h> // needed for RAPL
#include <pmt/ROCM.h> // needed for PMT
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include "kernals.h"

int main( int argc, char *argv[] )  {

    kernal kernal;
    kernal.name = "axpy";

    /* VERY DUMB Argument Parsers */
    kernal.size = parse_arguments(argc, argv);

    X_TYPE *d_sx; /* n is an array of N integers */
    X_TYPE *d_sy; /* n is an array of N integers */

    X_TYPE a = 2.0;
    // Allocate Host memory 
    X_TYPE* sx = (X_TYPE*)malloc(kernal.size* sizeof(X_TYPE));
    X_TYPE* sy = (X_TYPE*)malloc(kernal.size* sizeof(X_TYPE));

    // Allocate device memory 
    hipMalloc((void**)&d_sx, sizeof(X_TYPE) * kernal.size);
    hipMalloc((void**)&d_sy, sizeof(X_TYPE) * kernal.size);

    // THIS IS NEW !!!!!!!
    std::unique_ptr<pmt::PMT> GPUsensor = pmt::rocm::ROCM::Create();
    std::unique_ptr<pmt::PMT> CPUsensor = pmt::rapl::Rapl::Create();
    //Start the PMT "sensor"
    auto GPUstart = GPUsensor->Read(); 
    auto CPUstart = CPUsensor->Read(); 
    auto GPUend = GPUsensor->Read(); 
    auto CPUend = CPUsensor->Read(); 

    /* Simple saxpy */
    /*==============================*/
    if (true == simple)
    {
        kernal.algorithm = "simple_gpu";
        hipGetDevice(&kernal.gpuid);  

        int block_size = 512;
        int grid_size = ((kernal.size+ block_size) / block_size);
        
        do {
        //Start the PMT "sensor"
        auto GPUstart = GPUsensor->Read(); // READING the GPU via ROCM
        auto CPUstart = CPUsensor->Read(); // READING the CPU via RAPL
        
        // Transfer data from host to device memory
        hipMemcpy(d_sx, sx, sizeof(X_TYPE) * kernal.size, hipMemcpyHostToDevice);
        hipMemcpy(d_sy, sy, sizeof(X_TYPE) * kernal.size, hipMemcpyHostToDevice);

        gpu_axpy<<<grid_size,block_size>>>(kernal.size, a, d_sx, d_sy);

        hipMemcpy(sy, d_sy, sizeof(X_TYPE) * kernal.size, hipMemcpyDeviceToHost);

        //Start the PMT "sensor"
        auto GPUend = GPUsensor->Read();
        auto CPUend = CPUsensor->Read();

    kernal.rapl_times[kernal.N_runs] = pmt::PMT::seconds(CPUstart, CPUend);
    kernal.rapl_powers[kernal.N_runs] = pmt::PMT::watts(CPUstart, CPUend);
    kernal.rapl_energys[kernal.N_runs] = pmt::PMT::joules(CPUstart, CPUend);

    kernal.rocm_times[kernal.N_runs] = pmt::PMT::seconds(GPUstart, GPUend);
    kernal.rocm_powers[kernal.N_runs] = pmt::PMT::watts(GPUstart, GPUend);
    kernal.rocm_energys[kernal.N_runs] = pmt::PMT::joules(GPUstart, GPUend);
    kernal.N_runs ++;
    }while (kernal.time < kernal.max_time && kernal.N_runs < kernal.max_runs);
    kernal.calculate_stats();
  }
    kernal.print_pmt_rocm_info();
}
