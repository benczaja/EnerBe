#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h> // needed for ‘printf’
#include <stdlib.h> // needed for ‘RAND_MAX’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "helper.h" // local helper header to clean up code

#ifdef USE_DOUBLE
typedef double X_TYPE;
#else
typedef float X_TYPE;
#endif

void initialize_matrices(X_TYPE* A, X_TYPE* B, X_TYPE* C, int ROWS, int COLUMNS){
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

void simple_matrix_multiply(X_TYPE* D_A, X_TYPE* D_B, X_TYPE* D_C, int ROWS, int COLUMNS,
                            const sycl::nd_item<3> &item_ct1){

    int local_COLUMN = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
                int local_ROW =
                    item_ct1.get_local_id(1) +
                    item_ct1.get_group(1) * item_ct1.get_local_range(1);
                int local_index = local_COLUMN + local_ROW * ROWS; // Right now this only works for symetric matricies
		int tmp = 0;  
    
    if(local_ROW < ROWS && local_COLUMN < COLUMNS){
			for(int k=0; k<COLUMNS; k++){
				tmp += D_A[local_ROW * ROWS + k] * D_B[k * COLUMNS + local_COLUMN];
			}
			D_C[local_index] = tmp;
		}
}

int main(int argc, char *argv[]) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();

    printf("X_TYPE size is (%d) bytes \n",sizeof (X_TYPE));


  int ROWS;
  int COLUMNS;
  int N;
  clock_t t; // declare clock_t (long type)

  /* DUMB bools needed for the argument parsing logic */
  bool openmp = false;
  bool simple = true;
  bool sanity_check = false;
  
  /* VERY DUMB Argument Parsers */
  N = parse_arguments(argc, argv, &simple, &openmp, &sanity_check);
  ROWS = N;
  COLUMNS = N;

  /* declare the arrays...  better to do it as 1D arrays for CUDA */

  // First allocated them on the host (CPU)
    X_TYPE* A = (X_TYPE*)malloc((ROWS * COLUMNS) * sizeof(X_TYPE));
    X_TYPE* B = (X_TYPE*)malloc((ROWS * COLUMNS) * sizeof(X_TYPE));
    X_TYPE* C = (X_TYPE*)malloc((ROWS * COLUMNS) * sizeof(X_TYPE));

  // Then Allocate them on the GPUs
  X_TYPE* D_A;
  X_TYPE* D_B;
  X_TYPE* D_C;
  D_A = sycl::malloc_device<X_TYPE>((ROWS * COLUMNS), q_ct1);
  D_B = sycl::malloc_device<X_TYPE>((ROWS * COLUMNS), q_ct1);
  D_C = sycl::malloc_device<X_TYPE>((ROWS * COLUMNS), q_ct1);

  double start = omp_get_wtime();  

  initialize_matrices(A, B, C, ROWS, COLUMNS);
    
  double end = omp_get_wtime(); 
  printf("Init TIME: %f s\n",(end-start));


  /*======================================================================*/
  /*                START of Section of the code that matters!!!          */
  /*======================================================================*/

  /* Simple matrix multiplication */
  /*==============================*/
    int block_size = 512;
    int grid_size = ((ROWS + block_size) / block_size);
    
    t = clock(); // start the clock

    // Transfer data from host to device memory
    q_ct1.memcpy(D_A, A, sizeof(X_TYPE) * (ROWS * COLUMNS));
    q_ct1.memcpy(D_B, B, sizeof(X_TYPE) * (ROWS * COLUMNS));

    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                           simple_matrix_multiply(D_A, D_B, D_C, ROWS, COLUMNS,
                                                  item_ct1);
                       });

   // Transfer data from device to host memory
    q_ct1.memcpy(C, D_C, sizeof(X_TYPE) * (ROWS * COLUMNS)).wait();

    t = clock() - t; // stop the clock

    double time_taken = ((double)t)/CLOCKS_PER_SEC; // convert to seconds (and long to double)
    printf("GPU Compute Time: %f s\n",time_taken);

  /*======================================================================*/
  /*                 END of Section of the code that matters!!!           */
  /*======================================================================*/

 // Deallocate device memory
    dpct::dpct_free(D_A, q_ct1);
    dpct::dpct_free(D_B, q_ct1);
    dpct::dpct_free(D_C, q_ct1);

  // Deallocate host memory
  free(A);
  free(B);
  free(C);
}
