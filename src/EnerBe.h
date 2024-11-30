#include <ctype.h> // needed for isdigit
#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘EXIT_FAILURE’ 
#include <string.h> // needed for strcmp
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include <stdbool.h> // needed for bool usage
#include <omp.h> // needed for OpenMP 
#include <math.h>
#include <pmt.h> // needed for PMT

#ifdef USE_DOUBLE
typedef double X_TYPE;
#else
typedef float X_TYPE;
#endif
using namespace std;

class EnerBe {      
  
  public:

    std::string name;
    std::string algorithm;
    std::string perf_unit = "UNK";
    std::string gpu_vendor = "UNK";
    
    int omp_threads = omp_get_max_threads();
    int gpuid = 99;
    int N_gpus = 0;
    int MPI_ranks = 0;
    int N_runs = 0;
    int max_runs = 20;
    int size = 0;
    int measure_idx = 0;


    double start = 0.0;
    double end = 0.0;
    double max_time = 600.0;
    double time = 0.0;
    double time_var = 0.0;
    double time_std = 0.0;
    double perf = 0;

    // Devil horns
    double sign_of_the_beast = 0.0;

    double rapl_time = 0.0;
    double gpu_time = 0.0;

    double rapl_power = 0.0;
    double gpu_power = 0.0;

    double rapl_energy = 0.0;
    double gpu_energy = 0.0;

    double rapl_time_var = 0.0;
    double gpu_time_var = 0.0;
    double rapl_time_std = 0.0;
    double gpu_time_std = 0.0;

    double rapl_power_var = 0.0;
    double gpu_power_var = 0.0;
    double rapl_power_std = 0.0;
    double gpu_power_std = 0.0;

    double rapl_energy_var = 0.0;
    double gpu_energy_var = 0.0;

    double rapl_energy_std = 0.0;
    double gpu_energy_std = 0.0;

    #if defined(PMT_ENABLED) && defined(CUDA_ENABLED)
        //std::unique_ptr<pmt::PMT> GPUsensor = pmt::Create("NVML");
        //std::unique_ptr<pmt::PMT> CPUsensor = pmt::Create("Rapl");
        std::unique_ptr<pmt::PMT> GPUsensor = pmt::nvml::NVML::Create();
        std::unique_ptr<pmt::PMT> CPUsensor = pmt::rapl::Rapl::Create();
        
        pmt::State CPUstart = CPUsensor->Read();
        pmt::State CPUend = CPUsensor->Read();
        pmt::State GPUstart = GPUsensor->Read();
        pmt::State GPUend = GPUsensor->Read();
        
    #elif defined(PMT_ENABLED) && defined(HIP_ENABLED)

        std::unique_ptr<pmt::PMT> GPUsensor = pmt::rocm::ROCM::Create();
        std::unique_ptr<pmt::PMT> CPUsensor = pmt::rapl::Rapl::Create();
        
        pmt::State CPUstart = CPUsensor->Read();
        pmt::State CPUend = CPUsensor->Read();
        pmt::State GPUstart = GPUsensor->Read();
        pmt::State GPUend = GPUsensor->Read();

    #elif defined(PMT_ENABLED)
        //std::unique_ptr<pmt::PMT> CPUsensor = pmt::Create("Rapl");
        std::unique_ptr<pmt::PMT> CPUsensor = pmt::rapl::Rapl::Create();
        pmt::State CPUstart = CPUsensor->Read();
        pmt::State CPUend = CPUsensor->Read();
    #endif

    double times[20] =  {sign_of_the_beast};

    double rapl_times[20] =  {sign_of_the_beast};
    double rapl_powers[20] = {sign_of_the_beast};
    double rapl_energys[20] = {sign_of_the_beast};

    double gpu_times[20] =  {sign_of_the_beast};
    double gpu_powers[20] = {sign_of_the_beast};
    double gpu_energys[20] = {sign_of_the_beast};

    void print_info(){

        #if defined(PMT_ENABLED) && defined(CUDA_ENABLED)
            gpu_vendor = "NVML";
            print_pmt_gpu_info();
        #elif defined(PMT_ENABLED) && defined(HIP_ENABLED)
            gpu_vendor = "ROCM";
            print_pmt_gpu_info();
        #elif defined(PMT_ENABLED)
            print_pmt_rapl_info();
        #else
            print_basic_info();
        #endif
    }

    void print_basic_info() { 
        std::cout << "NAME: " << name << std::endl;
        std::cout << "ALGO: "<< algorithm << std::endl;
        std::cout << "PRECISION: "<< sizeof (X_TYPE) <<" bytes"<< std::endl;
        std::cout << "OMP_THREADS: "<< omp_threads << std::endl;
        std::cout << "MPI_RANKS: "<< MPI_ranks << std::endl;
        std::cout << "NGPUs: "<< N_gpus << std::endl;
        std::cout << "GPU ID: "<< gpuid << std::endl;
        std::cout << "SIZE: " << size << std::endl;
        std::cout << "PERF: " << perf << std::endl;
        std::cout << "PERF_UNIT: " << perf_unit << std::endl;
        std::cout << "NRUNS: " << N_runs << std::endl;
        std::cout << "TOTAL_TIME: " << time << " s"<< std::endl;
        std::cout << "TOTAL_TIME_var: " << time_var << " s^2"<< std::endl;
        std::cout << "TOTAL_TIME_std: " << time_std << " s"<< std::endl;
    } 

    void print_pmt_rapl_info() { 
        print_basic_info();
        std::cout << "(RAPL) CPU_TIME: " << rapl_time << " s"<< std::endl;
        std::cout << "(RAPL) CPU_TIME_var: " << rapl_time_var << " s^2"<< std::endl;
        std::cout << "(RAPL) CPU_TIME_std: " << rapl_time_std << " s"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS: " << rapl_power << " W" << std::endl;
        std::cout << "(RAPL) CPU_WATTS_var: " << rapl_power_var << " W^2" << std::endl;
        std::cout << "(RAPL) CPU_WATTS_std: " << rapl_power_std << " W" << std::endl;
        std::cout << "(RAPL) CPU_JOULES: " << rapl_energy << " J" << std::endl;
        std::cout << "(RAPL) CPU_JOULES_var: " << rapl_energy_var << " J^2" << std::endl;
        std::cout << "(RAPL) CPU_JOULES_std: " << rapl_energy_std << " J" << std::endl;
        std::cout << "TOTAL_WATTS: " << rapl_power + gpu_power << " W"<< std::endl;
        std::cout << "TOTAL_JOULES: " << rapl_energy + gpu_energy << " J"<< std::endl;

    } 

    void print_pmt_gpu_info() { 
        print_basic_info();
        std::cout << "(RAPL) CPU_TIME: " << rapl_time << " | (" << gpu_vendor << ") GPU_TIME: " << gpu_time << " s"<< std::endl;
        std::cout << "(RAPL) CPU_TIME_var: " << rapl_time_var << " | (" << gpu_vendor << ") GPU_TIME_var: " << gpu_time_var << " s^2"<< std::endl;
        std::cout << "(RAPL) CPU_TIME_std: " << rapl_time_std << " | (" << gpu_vendor << ") GPU_TIME_std: " << gpu_time_std << " s"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS: " << rapl_power << " | (" << gpu_vendor << ") GPU_WATTS: " << gpu_power << " W"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS_var: " << rapl_power_var << " | (" << gpu_vendor << ") GPU_WATTS_var: " << gpu_power_var << " W^2"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS_std: " << rapl_power_std << " | (" << gpu_vendor << ") GPU_WATTS_std: " << gpu_power_std << " W"<< std::endl;
        std::cout << "(RAPL) CPU_JOULES: " << rapl_energy << " | (" << gpu_vendor << ") GPU_JOULES: " << gpu_energy << " J"<< std::endl;
        std::cout << "(RAPL) CPU_JOULES_var: " << rapl_energy_var << " | (" << gpu_vendor << ") GPU_JOULES_var: " << gpu_energy_var << " J^2"<< std::endl;
        std::cout << "(RAPL) CPU_JOULES_std: " << rapl_energy_std << " | (" << gpu_vendor << ") GPU_JOULES_std: " << gpu_energy_std << " J"<< std::endl;
        std::cout << "TOTAL_WATTS: " << rapl_power + gpu_power << " W"<< std::endl;
        std::cout << "TOTAL_JOULES: " << rapl_energy + gpu_energy << " J"<< std::endl;
    } 

    void measure(){

        if (measure_idx == 0 ){
            #if defined(PMT_ENABLED) && defined(CUDA_ENABLED)
                CPUstart = CPUsensor->Read();
                GPUstart = GPUsensor->Read();
            #elif defined(PMT_ENABLED) && defined(HIP_ENABLED)
                CPUstart = CPUsensor->Read();
                GPUstart = GPUsensor->Read();
            #elif defined(PMT_ENABLED)
                CPUstart = CPUsensor->Read();
            #else
                start = omp_get_wtime();
            #endif

            measure_idx = 1;

        }else if (measure_idx == 1) {


            #if defined(PMT_ENABLED) && defined(CUDA_ENABLED)
                GPUend = GPUsensor->Read();
                CPUend = CPUsensor->Read();

                rapl_times[N_runs] = pmt::PMT::seconds(CPUstart, CPUend);
                rapl_powers[N_runs] = pmt::PMT::watts(CPUstart, CPUend);
                rapl_energys[N_runs] = pmt::PMT::joules(CPUstart, CPUend);

                gpu_times[N_runs] = pmt::PMT::seconds(GPUstart, GPUend);
                gpu_powers[N_runs] = pmt::PMT::watts(GPUstart, GPUend);
                gpu_energys[N_runs] = pmt::PMT::joules(GPUstart, GPUend);

            #elif defined(PMT_ENABLED) && defined(HIP_ENABLED)
                GPUend = GPUsensor->Read();
                CPUend = CPUsensor->Read();

                rapl_times[N_runs] = pmt::PMT::seconds(CPUstart, CPUend);
                rapl_powers[N_runs] = pmt::PMT::watts(CPUstart, CPUend);
                rapl_energys[N_runs] = pmt::PMT::joules(CPUstart, CPUend);

                gpu_times[N_runs] = pmt::PMT::seconds(GPUstart, GPUend);
                gpu_powers[N_runs] = pmt::PMT::watts(GPUstart, GPUend);
                gpu_energys[N_runs] = pmt::PMT::joules(GPUstart, GPUend);

            #elif defined(PMT_ENABLED)
                CPUend = CPUsensor->Read();
                rapl_times[N_runs] = pmt::PMT::seconds(CPUstart, CPUend);
                rapl_powers[N_runs] = pmt::PMT::watts(CPUstart, CPUend);
                rapl_energys[N_runs] = pmt::PMT::joules(CPUstart, CPUend);
            #else
                end = omp_get_wtime();
                times[N_runs] += end - start;
            #endif

            measure_idx = 0;
        }


    }

    void calculate_stats(){
        
        double sum_time = 0.0;
        double sum_rapl_time = 0.0;
        double sum_gpu_time = 0.0;

        double sum_energy = 0.0;
        double sum_rapl_energy = 0.0;
        double sum_gpu_energy = 0.0;

        double sum_power = 0.0;
        double sum_rapl_power = 0.0;
        double sum_gpu_power = 0.0;

        for(int i=0; i<N_runs ;i++){
            sum_time += times[i];

            sum_rapl_time   += rapl_times[i];
            sum_rapl_power  += rapl_powers[i];
            sum_rapl_energy += rapl_energys[i];

            sum_gpu_time   += gpu_times[i];
            sum_gpu_power  += gpu_powers[i];
            sum_gpu_energy += gpu_energys[i];

        }

        rapl_time = sum_rapl_time/double(N_runs);
        gpu_time = sum_gpu_time/double(N_runs);

        time = (rapl_time + gpu_time); // Maybe this is wrong??

        rapl_power = sum_rapl_power/double(N_runs);
        gpu_power = sum_gpu_power/double(N_runs);

        rapl_energy = sum_rapl_energy/double(N_runs);
        gpu_energy = sum_gpu_energy/double(N_runs);

        double values = 0;
        double rapl_time_values = 0;
        double gpu_time_values = 0;

        double rapl_power_values = 0;
        double gpu_power_values = 0;

        double rapl_energy_values = 0;
        double gpu_energy_values = 0;

        for(int i = 0; i < N_runs; i++) {
            values += pow(times[i] - time, 2);

            rapl_time_values += pow(rapl_times[i] - rapl_time, 2);
            gpu_time_values += pow(gpu_times[i] - gpu_time, 2);

            rapl_power_values += pow(rapl_powers[i] - rapl_power, 2);
            gpu_power_values += pow(gpu_powers[i] - gpu_power, 2);

            rapl_energy_values += pow(rapl_energys[i] - rapl_energy, 2);
            gpu_energy_values += pow(gpu_energys[i] - gpu_energy, 2);
        }
    
        // variance is the square of standard deviation

        rapl_time_var = rapl_time_values /double(N_runs);
        gpu_time_var = gpu_time_values /double(N_runs);

        rapl_power_var = rapl_power_values /double(N_runs);
        gpu_power_var = gpu_power_values /double(N_runs);

        rapl_energy_var = rapl_energy_values /double(N_runs);
        gpu_energy_var = gpu_energy_values /double(N_runs);

        time_var = rapl_time_var + gpu_time_var;

        // calculating standard deviation by finding square root
        // of variance
        time_std = sqrt(time_var);

        rapl_time_std   = sqrt(rapl_time_var);
        gpu_time_std   = sqrt(gpu_time_var);

        rapl_power_std  = sqrt(rapl_power_var);
        gpu_power_std  = sqrt(gpu_power_var);

        rapl_energy_std = sqrt(rapl_energy_var);
        gpu_energy_std = sqrt(gpu_energy_var);

        
        if (name == "xgemm")
            {
                perf = size * size * sizeof (X_TYPE) * 8.0 / (time); // FLOP/s
                perf_unit = "FLOPs";
            }

    }

};



