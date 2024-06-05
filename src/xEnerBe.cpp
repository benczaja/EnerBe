#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘RAND_MAX’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "mesh.hpp"
#include "argparser.h"
//#include "kernals.h"

using namespace std;

int main( int argc, char *argv[] )  {
    
  Mesh2D_t Mesh2D;
  MM_t MM(Mesh2D);


  parse_arguments(argc, argv, MM.size, MM.algorithm, MM.name);

  MM.InitializeMatrix();

  MM.print_info();

  //cout << MM.A[0] <<endl;
  
  /* initialize the arrays */
  //kernal.initialize_matrix_1D();//(kernal.A, kernal.B, kernal.C, kernal.size, kernal.size);

}

///* Simple matrix multiplication */
///*==============================*/
//if (kernal.name == "xgemm" && kernal.algorithm == "simple")
//{
//  std::cout<<kernal.A[0]<<std::endl;
//  clock_t t; // declare clock_t (long type)
//  kernal.omp_threads = 1;
//  do {
//    kernal.start = double(clock());
//    kernal.simple_matrix_multiply(kernal.size, kernal.size);
//    kernal.end = double(clock());
//    kernal.times[kernal.N_runs] =  (kernal.end - kernal.start)/CLOCKS_PER_SEC;
//    kernal.N_runs ++;
//  }while (kernal.time < kernal.max_time && kernal.N_runs < kernal.max_runs);
//  kernal.calculate_stats();
//}
///* OpenMP parallel matrix multiplication */
///*=======================================*/
//if (kernal.name == "xgemm" && kernal.algorithm == "openmp")
//{
//  // omp_get_wtime needed here because clock will sum up time for all threads
//  do {
//    kernal.start = omp_get_wtime();  
//    kernal.openmp_matrix_multiply(kernal.size, kernal.size);
//    kernal.end = omp_get_wtime(); 
//    kernal.times[kernal.N_runs] += kernal.end - kernal.start;
//    kernal.N_runs ++;
//  }while (kernal.time < kernal.max_time && kernal.N_runs < kernal.max_runs);
//  kernal.calculate_stats();
//}
//kernal.print_info();
//
///* deallocate the arrays */
//
//