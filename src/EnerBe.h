#include <ctype.h> // needed for isdigit
#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘EXIT_FAILURE’ 
#include <string.h> // needed for strcmp
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include <stdbool.h> // needed for bool usage
#include <omp.h> // needed for OpenMP 
#include <math.h>

#ifdef USE_DOUBLE
typedef double X_TYPE;
#else
typedef float X_TYPE;
#endif

class EnerBe {      
  
  public:

    std::string name;
    std::string algorithm;
    
    int omp_threads = omp_get_max_threads();
    int gpuid = 99;
    int N_gpus = 0;
    int MPI_ranks = 0;
    int N_runs = 0;
    int max_runs = 20;
    int size = 0;


    double start = 0.0;
    double end = 0.0;
    double max_time = 600.0;
    double time = 0.0;
    double time_var = 0.0;
    double time_std = 0.0;

    // Devil horns
    double sign_of_the_beast = 0.0;

    double rapl_time = 0.0;
    double nvml_time = 0.0;
    double rocm_time = 0.0;

    double rapl_power = 0.0;
    double nvml_power = 0.0;
    double rocm_power = 0.0;

    double rapl_energy = 0.0;
    double nvml_energy = 0.0;
    double rocm_energy = 0.0;

    double rapl_time_var = 0.0;
    double nvml_time_var = 0.0;
    double rocm_time_var = 0.0;
    double rapl_time_std = 0.0;
    double nvml_time_std = 0.0;
    double rocm_time_std = 0.0;

    double rapl_power_var = 0.0;
    double nvml_power_var = 0.0;
    double rocm_power_var = 0.0;
    double rapl_power_std = 0.0;
    double nvml_power_std = 0.0;
    double rocm_power_std = 0.0;

    double rapl_energy_var = 0.0;
    double nvml_energy_var = 0.0;
    double rocm_energy_var = 0.0;

    double rapl_energy_std = 0.0;
    double nvml_energy_std = 0.0;
    double rocm_energy_std = 0.0;



    double times[20] =  {sign_of_the_beast};

    double rapl_times[20] =  {sign_of_the_beast};
    double rapl_powers[20] = {sign_of_the_beast};
    double rapl_energys[20] = {sign_of_the_beast};

    double nvml_times[20] =  {sign_of_the_beast};
    double nvml_powers[20] = {sign_of_the_beast};
    double nvml_energys[20] = {sign_of_the_beast};

    double rocm_times[20] =  {sign_of_the_beast};
    double rocm_powers[20] = {sign_of_the_beast};
    double rocm_energys[20] = {sign_of_the_beast};

    void print_info() { 
        std::cout << "NAME: " << name << std::endl;
        std::cout << "ALGO: "<< algorithm << std::endl;
        std::cout << "PRECISION: "<< sizeof (X_TYPE) <<" bytes"<< std::endl;
        std::cout << "OMP_THREADS: "<< omp_threads << std::endl;
        std::cout << "MPI_RANKS: "<< MPI_ranks << std::endl;
        std::cout << "NGPUs: "<< N_gpus << std::endl;
        std::cout << "GPU ID: "<< gpuid << std::endl;
        std::cout << "SIZE: " << size << std::endl;
        std::cout << "TOTAL_TIME: " << time << " s"<< std::endl;
        std::cout << "TOTAL_TIME_var: " << time_var << " s^2"<< std::endl;
        std::cout << "TOTAL_TIME_std: " << time_std << " s"<< std::endl;
        std::cout << "NRUNS: " << N_runs << std::endl;
    } 

    void print_pmt_rapl_info() { 
        std::cout << "NAME: " << name << std::endl;
        std::cout << "ALGO: "<< algorithm << std::endl;
        std::cout << "PRECISION: "<< sizeof (X_TYPE) <<" bytes"<< std::endl;
        std::cout << "OMP_THREADS: "<< omp_threads << std::endl;
        std::cout << "MPI_RANKS: "<< MPI_ranks << std::endl;
        std::cout << "NGPUs: "<< N_gpus << std::endl;
        std::cout << "GPU ID: "<< gpuid << std::endl;
        std::cout << "SIZE: " << size << std::endl;
        std::cout << "(RAPL) CPU_TIME: " << rapl_time << " s"<< std::endl;
        std::cout << "(RAPL) CPU_TIME_var: " << rapl_time_var << " s^2"<< std::endl;
        std::cout << "(RAPL) CPU_TIME_std: " << rapl_time_std << " s"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS: " << rapl_power << " W" << std::endl;
        std::cout << "(RAPL) CPU_WATTS_var: " << rapl_power_var << " W^2" << std::endl;
        std::cout << "(RAPL) CPU_WATTS_std: " << rapl_power_std << " W" << std::endl;
        std::cout << "(RAPL) CPU_JOULES: " << rapl_energy << " J" << std::endl;
        std::cout << "(RAPL) CPU_JOULES_var: " << rapl_energy_var << " J^2" << std::endl;
        std::cout << "(RAPL) CPU_JOULES_std: " << rapl_energy_std << " J" << std::endl;
        std::cout << "NRUNS: " << N_runs << std::endl;

    } 

    void print_pmt_nvml_info() { 
        std::cout << "NAME: " << name << std::endl;
        std::cout << "ALGO: "<< algorithm << std::endl;
        std::cout << "PRECISION: "<< sizeof (X_TYPE) <<" bytes"<< std::endl;
        std::cout << "OMP_THREADS: "<< omp_threads << std::endl;
        std::cout << "MPI_RANKS: "<< MPI_ranks << std::endl;
        std::cout << "NGPUs: "<< N_gpus << std::endl;
        std::cout << "GPU ID: "<< gpuid << std::endl;
        std::cout << "SIZE: " << size <<std::endl;
        std::cout << "(RAPL) CPU_TIME: " << rapl_time << " | (NVML) GPU_TIME: " << nvml_time << " s"<< std::endl;
        std::cout << "(RAPL) CPU_TIME_var: " << rapl_time_var << " | (NVML) GPU_TIME_var: " << nvml_time_var << " s^2"<< std::endl;
        std::cout << "(RAPL) CPU_TIME_std: " << rapl_time_std << " | (NVML) GPU_TIME_std: " << nvml_time_std << " s"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS: " << rapl_power << " | (NVML) GPU_WATTS: " << nvml_power << " W"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS_var: " << rapl_power_var << " | (NVML) GPU_WATTS_var: " << nvml_power_var << " W^2"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS_std: " << rapl_power_std << " | (NVML) GPU_WATTS_std: " << nvml_power_std << " W"<< std::endl;
        std::cout << "(RAPL) CPU_JOULES: " << rapl_energy << " | (NVML) GPU_JOULES: " << nvml_energy << " J"<< std::endl;
        std::cout << "(RAPL) CPU_JOULES_var: " << rapl_energy_var << " | (NVML) GPU_JOULES_var: " << nvml_energy_var << " J^2"<< std::endl;
        std::cout << "(RAPL) CPU_JOULES_std: " << rapl_energy_std << " | (NVML) GPU_JOULES_std: " << nvml_energy_std << " J"<< std::endl;
        std::cout << "TOTAL_TIME: " << rapl_time + nvml_time << " s"<< std::endl;
        std::cout << "TOTAL_WATTS: " << rapl_power + nvml_power << " W"<< std::endl;
        std::cout << "TOTAL_JOULES: " << rapl_energy + nvml_energy << " J"<< std::endl;
        std::cout << "NRUNS: " << N_runs << std::endl;
    } 

    void print_pmt_rocm_info() { 
        std::cout << "NAME: " << name << std::endl;
        std::cout << "ALGO: "<< algorithm << std::endl;
        std::cout << "PRECISION: "<< sizeof (X_TYPE) <<" bytes"<< std::endl;
        std::cout << "OMP_THREADS: "<< omp_threads << std::endl;
        std::cout << "MPI_RANKS: "<< MPI_ranks << std::endl;
        std::cout << "NGPUs: "<< N_gpus << std::endl;
        std::cout << "GPU ID: "<< gpuid << std::endl;
        std::cout << "SIZE: " << size <<std::endl;
        std::cout << "(RAPL) CPU_TIME: " << rapl_time << " | (ROCM) GPU_TIME: " << rocm_time << " s"<< std::endl;
        std::cout << "(RAPL) CPU_TIME_var: " << rapl_time_var << " | (ROCM) GPU_TIME_var: " << rocm_time_var << " s^2"<< std::endl;
        std::cout << "(RAPL) CPU_TIME_std: " << rapl_time_std << " | (ROCM) GPU_TIME_std: " << rocm_time_std << " s"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS: " << rapl_power << " | (ROCM) GPU_WATTS: " << rocm_power << " W"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS_var: " << rapl_power_var << " | (ROCM) GPU_WATTS_var: " << rocm_power_var << " W^2"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS_std: " << rapl_power_std << " | (ROCM) GPU_WATTS_std: " << rocm_power_std << " W"<< std::endl;
        std::cout << "(RAPL) CPU_JOULES: " << rapl_energy << " | (ROCM) GPU_JOULES: " << rocm_energy << " J"<< std::endl;
        std::cout << "(RAPL) CPU_JOULES_var: " << rapl_energy_var << " | (ROCM) GPU_JOULES_var: " << rocm_energy_var << " J^2"<< std::endl;
        std::cout << "(RAPL) CPU_JOULES_std: " << rapl_energy_std << " | (ROCM) GPU_JOULES_std: " << rocm_energy_std << " J"<< std::endl;
        std::cout << "Total TIME: " << rapl_time + rocm_time << " s"<< std::endl;
        std::cout << "Total WATTS: " << rapl_power + rocm_power << " W"<< std::endl;
        std::cout << "Total JOULES: " << rapl_energy + rocm_energy << " J"<< std::endl;
        std::cout << "NRUNS: " << N_runs << std::endl;
    } 

    void calculate_stats(){
        
        double sum_time = 0.0;
        double sum_rapl_time = 0.0;
        double sum_nvml_time = 0.0;
        double sum_rocm_time = 0.0;

        double sum_energy = 0.0;
        double sum_rapl_energy = 0.0;
        double sum_nvml_energy = 0.0;
        double sum_rocm_energy = 0.0;

        double sum_power = 0.0;
        double sum_rapl_power = 0.0;
        double sum_nvml_power = 0.0;
        double sum_rocm_power = 0.0;

        for(int i=0; i<N_runs ;i++){
            sum_time += times[i];

            sum_rapl_time   += rapl_times[i];
            sum_rapl_power  += rapl_powers[i];
            sum_rapl_energy += rapl_energys[i];

            sum_nvml_time   += nvml_times[i];
            sum_nvml_power  += nvml_powers[i];
            sum_nvml_energy += nvml_energys[i];

            sum_rocm_time   += rocm_times[i];
            sum_rocm_power  += rocm_powers[i];
            sum_rocm_energy += rocm_energys[i];

        }
        time = sum_time/double(N_runs);

        rapl_time = sum_rapl_time/double(N_runs);
        nvml_time = sum_nvml_time/double(N_runs);
        rocm_time = sum_rocm_time/double(N_runs);

        rapl_power = sum_rapl_power/double(N_runs);
        nvml_power = sum_nvml_power/double(N_runs);
        rocm_power = sum_rocm_power/double(N_runs);

        rapl_energy = sum_rapl_energy/double(N_runs);
        nvml_energy = sum_nvml_energy/double(N_runs);
        rocm_energy = sum_rocm_energy/double(N_runs);

        double values = 0;
        double rapl_time_values = 0;
        double nvml_time_values = 0;
        double rocm_time_values = 0;

        double rapl_power_values = 0;
        double nvml_power_values = 0;
        double rocm_power_values = 0;

        double rapl_energy_values = 0;
        double nvml_energy_values = 0;
        double rocm_energy_values = 0;
        for(int i = 0; i < N_runs; i++) {
            values += pow(times[i] - time, 2);

            rapl_time_values += pow(rapl_times[i] - rapl_time, 2);
            nvml_time_values += pow(nvml_times[i] - nvml_time, 2);
            rocm_time_values += pow(rocm_times[i] - rocm_time, 2);

            rapl_power_values += pow(rapl_powers[i] - rapl_power, 2);
            nvml_power_values += pow(nvml_powers[i] - nvml_power, 2);
            rocm_power_values += pow(rocm_powers[i] - rocm_power, 2);

            rapl_energy_values += pow(rapl_energys[i] - rapl_energy, 2);
            nvml_energy_values += pow(nvml_energys[i] - nvml_energy, 2);
            rocm_energy_values += pow(rocm_energys[i] - rocm_energy, 2);
        }
    
        // variance is the square of standard deviation
        time_var = values /double(N_runs);

        rapl_time_var = rapl_time_values /double(N_runs);
        nvml_time_var = nvml_time_values /double(N_runs);
        rocm_time_var = rocm_time_values /double(N_runs);

        rapl_power_var = rapl_power_values /double(N_runs);
        nvml_power_var = nvml_power_values /double(N_runs);
        rocm_power_var = rocm_power_values /double(N_runs);

        rapl_energy_var = rapl_energy_values /double(N_runs);
        nvml_energy_var = nvml_energy_values /double(N_runs);
        rocm_energy_var = rocm_energy_values /double(N_runs);
        // calculating standard deviation by finding square root
        // of variance
        time_std = sqrt(time_var);

        rapl_time_std   = sqrt(rapl_time_var);
        nvml_time_std   = sqrt(nvml_time_var);
        rocm_time_std   = sqrt(rocm_time_var);
        rapl_power_std  = sqrt(rapl_power_var);
        nvml_power_std  = sqrt(nvml_power_var);
        rocm_power_std  = sqrt(rocm_power_var);
        rapl_energy_std = sqrt(rapl_energy_var);
        nvml_energy_std = sqrt(nvml_energy_var);
        rocm_energy_std = sqrt(rocm_energy_var);
    }

};



