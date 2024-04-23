#include <ctype.h> // needed for isdigit
#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘EXIT_FAILURE’ 
#include <string.h> // needed for strcmp
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include <stdbool.h> // needed for bool usage
#include <omp.h> // needed for OpenMP 

#ifdef USE_DOUBLE
typedef double X_TYPE;
#else
typedef float X_TYPE;
#endif


class kernal {      
  public:

    std::string name;
    std::string algorithm;
    
    double size = sizeof(X_TYPE);
    double time = 0.0;
    double rapl_time = 0.0;
    double nvml_time = 0.0;
    double rocm_time = 0.0;

    double rapl_power = 0.0;
    double nvml_power = 0.0;
    double rocm_power = 0.0;

    double rapl_energy = 0.0;
    double nvml_energy = 0.0;
    double rocm_energy = 0.0;

    void print_info() { 
        std::cout << "NAME: " << name << std::endl;
        std::cout << "ALGO: "<< algorithm << std::endl;
        std::cout << "PRECISION: "<< sizeof (X_TYPE) <<" bytes"<< std::endl;
        std::cout << "OMP_THREADS: "<< omp_get_max_threads() << std::endl;
        std::cout << "SIZE: " << size << std::endl;
        std::cout << "TIME: " << time << " s"<< std::endl;
    } 

    void print_pmt_rapl_info() { 
        std::cout << "NAME: " << name << std::endl;
        std::cout << "ALGO: "<< algorithm << std::endl;
        std::cout << "PRECISION: "<< sizeof (X_TYPE) <<" bytes"<< std::endl;
        std::cout << "OMP_THREADS: "<< omp_get_max_threads() << std::endl;
        std::cout << "SIZE: " << size << std::endl;
        std::cout << "(RAPL) CPU_TIME: " << rapl_time << " s"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS: " << rapl_power << " W" << std::endl;
        std::cout << "(RAPL) CPU_JOULES: " << rapl_energy << " J" << std::endl;
    } 

    void print_pmt_nvml_info() { 
        std::cout << "NAME: " << name << std::endl;
        std::cout << "ALGO: "<< algorithm << std::endl;
        std::cout << "PRECISION: "<< sizeof (X_TYPE) <<" bytes"<< std::endl;
        std::cout << "OMP_THREADS: "<< omp_get_max_threads() << std::endl;
        std::cout << "SIZE: " << size <<std::endl;
        std::cout << "(RAPL) CPU_TIME: " << rapl_time << " s"<< std::endl;
        std::cout << "(RAPL) CPU_WATTS: " << rapl_power << " W" << std::endl;
        std::cout << "(RAPL) CPU_JOULES: " << rapl_energy << " J" << std::endl;
    } 
};


/* DUMB bools needed for the argument parsing logic */
bool openmp = false;
bool simple = false;
bool sanity_check = false;

void initialize_matrix_1D(X_TYPE* A, X_TYPE* B, X_TYPE* C, int ROWS, int COLUMNS){
    // Do this in Parallel with OpenMP
    // Needs a seperate seed per thread as rand() is obtaining a mutex and therefore locking each thread.
    unsigned int globalSeed = clock();  
    #pragma omp parallel for
    for (int i = 0; i < ROWS * COLUMNS; i++)
        {
          unsigned int randomState = i ^ globalSeed;
          A[i] = (X_TYPE) rand_r(&randomState) / RAND_MAX;
          B[i] = (X_TYPE) rand_r(&randomState) / RAND_MAX;
          C[i] = 0.0 ;
        }
}

void initialize_matrix_2D(X_TYPE** A, X_TYPE** B, X_TYPE** C, int ROWS, int COLUMNS){
    for (int i = 0 ; i < ROWS ; i++)
    {
        for (int j = 0 ; j < COLUMNS ; j++)
        {
            A[i][j] = (X_TYPE) rand() / RAND_MAX ;
            B[i][j] = (X_TYPE) rand() / RAND_MAX ;
            C[i][j] = 0.0 ;
        }
    }
}





void print_saxpy_usage()
{
    fprintf(stderr, "saxpy (array size) [-s|-p|-h]\n");
    fprintf(stderr, "\t      Invoke simple implementation of Saxpy (Single precision A X plus Y)\n");
    fprintf(stderr, "\t-s    Invoke simple implementation of Saxpy (Single precision A X plus Y)\n");
    fprintf(stderr, "\t-p    Invoke parallel (OpenMP) implementation of Saxpy (Single precision A X plus Y)\n");
    fprintf(stderr, "\t-h    Display help\n");
}

void print_xgemm_usage()
{
    fprintf(stderr, "xgemm (matrix size) [-s|-p|-h]\n");
    fprintf(stderr, "\t      Invoke simple implementation of matrix multiplication\n");
    fprintf(stderr, "\t-s    Invoke simple implementation of matrix multiplication\n");
    fprintf(stderr, "\t-p    Invoke parallel (OpenMP) implementation of matrix multiplication\n");
    fprintf(stderr, "\t-h    Display help\n");
}

bool isNumber(char number[])
{
    int i = 0;

    //checking for negative numbers
    if (number[0] == '-')
        i = 1;
    for (; number[i] != 0; i++)
    {
        //if (number[i] > '9' || number[i] < '0')
        if (!isdigit(number[i]))
            return false;
    }
    return true;
}

int parse_arguments(size_t count, char*  args[], bool *simple, bool *openmp, bool *sanity_check) {
    int N;
    if (count == 1){
        printf("I need an alogrithm and problem size as an argument.\nSee what I accept: ./dgemm -h \n");

        exit (1);
    }
    for(int i=0; i<count; ++i)
        {   
            if (! strcmp("--simple", args[i]))
            {
                *simple = true;
            }
            else if (! strcmp("--openmp", args[i]))
            {
                *openmp = true;
                *simple = false;
            }
            else if (!strcmp("-h", args[i]))
            {
                if (strstr(args[0],"saxpy"))
                {
                print_saxpy_usage();
                }
                else if (strstr(args[0],"gemm"))
                {
                print_xgemm_usage();
                }
                exit (1);
            }
            else if (isNumber(args[i]))
            {
              sscanf(args[i],"%d", &N);
              *sanity_check = true;
            }
        }
        /* Dumb logic to make sure an array size was passed */
    if (! sanity_check){
        printf("I need an alogrithm and problem size as an argument.\nSee what I accept: ./dgemm -h \n");
        exit (1);

    }
    return (N);
}