#include "mesh.hpp"
#include <../../OpenBLAS/include/cblas.h>
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include "cublas_v2.h"
#endif

using namespace std;

#ifdef CUDA_ENABLED
//////////////////////////////////Call Checks///////////////////
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    const cublasStatus_t err = call;                                                 \
    if (err != CUBLAS_STATUS_SUCCESS)                                           \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__, __LINE__); \
        printf("Reason: %s\n", cublasGetStatusName(err));                      \
        exit(1);                                                               \
    }                                                                          \
}
#endif


MM_t::MM_t(Mesh2D_t& Mesh2D_):
  Mesh2D(Mesh2D_)
{
    Initialize_symmetric_matricies_ABC();
    run();
}


MM_t::~MM_t() {
  free(Mesh2D.A);
  free(Mesh2D.B);
  free(Mesh2D.C);

  #ifdef CUDA_ENABLED
    CHECK(cudaFree(Mesh2D.D_A));
  	CHECK(cudaFree(Mesh2D.D_B));
  	CHECK(cudaFree(Mesh2D.D_C));
  #endif

}



// Allocated A, B, C matrixies
void MM_t::Initialize_symmetric_matricies_ABC() 
{
    Mesh2D.size = size;
    
    Mesh2D.A = (X_TYPE *) malloc((Mesh2D.size * Mesh2D.size)*sizeof(X_TYPE));
    Mesh2D.B = (X_TYPE *) malloc((Mesh2D.size * Mesh2D.size)*sizeof(X_TYPE));
    Mesh2D.C = (X_TYPE *) malloc((Mesh2D.size * Mesh2D.size)*sizeof(X_TYPE));

    #ifdef CUDA_ENABLED
        CHECK(cudaMalloc((void**)&Mesh2D.D_A, sizeof( X_TYPE ) * (Mesh2D.size * Mesh2D.size)));
        CHECK(cudaMalloc((void**)&Mesh2D.D_B, sizeof( X_TYPE ) * (Mesh2D.size * Mesh2D.size)));
        CHECK(cudaMalloc((void**)&Mesh2D.D_C, sizeof( X_TYPE ) * (Mesh2D.size * Mesh2D.size)));
    #endif

    unsigned int globalSeed = clock();  

    for (int i = 0; i < (Mesh2D.size * Mesh2D.size); i++)
    {
        unsigned int randomState = i ^ globalSeed;
        Mesh2D.A[i] = (X_TYPE) rand_r(&randomState) / RAND_MAX;
        Mesh2D.B[i] = (X_TYPE) rand_r(&randomState) / RAND_MAX;
        Mesh2D.C[i] = 0.0;
    }

}

void MM_t::PrintElement(int i)
{
    cout << "Mesh2D.A["<<i<<"]: " <<Mesh2D.A[i]<<endl;
}

void MM_t::simple_matrix_multiply(int ROWS, int COLUMNS)
{
    for(int i=0;i<ROWS;i++)
    {
        for(int j=0;j<COLUMNS;j++)
        {
            for(int k=0;k<COLUMNS;k++)
            {
                // ROWS and COLS might be messed up here
                Mesh2D.C[i * COLUMNS + j] += Mesh2D.A[i * ROWS + k] * Mesh2D.B[k * COLUMNS + j];
            }
        }
    }
}


void MM_t::openmp_matrix_multiply(int ROWS, int COLUMNS)
{
    #pragma omp parallel for
    for(int i=0;i<ROWS;i++)
    {
        for(int j=0;j<COLUMNS;j++)
        {
            for(int k=0;k<COLUMNS;k++)
            {
                // ROWS and COLS might be messed up here
                Mesh2D.C[i * COLUMNS + j] += Mesh2D.A[i * ROWS + k] * Mesh2D.B[k * COLUMNS + j];
            }
        }
    }
}

void MM_t::cblas_matrix_multiply(int ROWS, int COLUMNS)
{
    #ifdef USE_DOUBLE
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ROWS, ROWS, COLUMNS, 1.0, Mesh2D.A, COLUMNS, Mesh2D.B, COLUMNS, 1.0, Mesh2D.C, COLUMNS);
    #else 
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,  ROWS, ROWS, COLUMNS, 1.0, Mesh2D.A, COLUMNS, Mesh2D.B, COLUMNS, 1.0, Mesh2D.C, COLUMNS);
    #endif

}




void MM_t::simple_jacobi(X_TYPE* A, X_TYPE* B, X_TYPE* C, X_TYPE* Ctmp, int ROWS, int COLUMNS)
{
  int itr;
  int row, col;
  int MAX_ITERATIONS = 20000;
  X_TYPE CONVERGENCE_THRESHOLD = 0.0001;
  X_TYPE dot;
  X_TYPE diff;
  X_TYPE sqdiff;
  X_TYPE* ptrtmp;
  // Loop until converged or maximum iterations reached
  itr = 0;
  do
  {
    // Perfom Jacobi iteration
    for (row = 0; row < ROWS; row++)
    {
      dot = 0.0;
      for (col = 0; col < COLUMNS; col++)
      {
        if (row != col){
          dot += A[row + col*ROWS] * C[col];
        }
      }
      Ctmp[row] = (B[row] - dot) / A[row + row*ROWS];
    }

    // Swap pointers
    ptrtmp = C;
    C      = Ctmp;
    Ctmp   = ptrtmp;

    // Check for convergence
    sqdiff = 0.0;
    for (row = 0; row < ROWS; row++)
    {
      diff    = Ctmp[row] - C[row];      
      sqdiff += diff * diff;
    }

    itr++;
  } while ((itr < MAX_ITERATIONS) && (sqrt(sqdiff) > CONVERGENCE_THRESHOLD));
  std::cout<<"Jocobi solved in: " << itr <<" Iterations"<< std::endl;
}


void MM_t::openmp_jacobi(X_TYPE* A, X_TYPE* B, X_TYPE* C, X_TYPE* Ctmp, int ROWS, int COLUMNS)
{
  int itr;
  int row, col;
  int MAX_ITERATIONS = 20000;
  X_TYPE CONVERGENCE_THRESHOLD = 0.0001;
  X_TYPE dot;
  X_TYPE diff;
  X_TYPE sqdiff;
  X_TYPE* ptrtmp;
  // Loop until converged or maximum iterations reached
  itr = 0;
  do
  {
    // Perfom Jacobi iteration
    #pragma omp parallel for
    for (row = 0; row < ROWS; row++)
    {
      dot = 0.0;
      for (col = 0; col < COLUMNS; col++)
      {
        if (row != col){
          dot += A[row + col*ROWS] * C[col];
        }
      }
      Ctmp[row] = (B[row] - dot) / A[row + row*ROWS];
    }

    #pragma omp barrier

    // Swap pointers
    ptrtmp = C;
    C      = Ctmp;
    Ctmp   = ptrtmp;

    // Check for convergence
    sqdiff = 0.0;
    #pragma omp parallel for
    for (row = 0; row < ROWS; row++)
    {
      diff    = Ctmp[row] - C[row];      
      sqdiff += diff * diff;
    }

    itr++;
  } while ((itr < MAX_ITERATIONS) && (sqrt(sqdiff) > CONVERGENCE_THRESHOLD));
  std::cout<<"Jocobi solved in: " << itr <<" Iterations"<< std::endl;
}






#ifdef CUDA_ENABLED
    __global__ void gpu_thread_matrix_multiply(X_TYPE* D_A, X_TYPE* D_B, X_TYPE* D_C,int ROWS, int COLUMNS){
    // This will solve the matmul by assigning each GPU thread to a single site on the result matrix C
    // Each GPU thread has a local index in its CUDA thread block
    
    int local_COLUMN = threadIdx.x + blockIdx.x * blockDim.x;
	int local_ROW = threadIdx.y + blockIdx.y * blockDim.y;
	int local_index = local_COLUMN + local_ROW * ROWS; // Right now this only works for symetric matricies
	int tmp = 0;  
    
    if(local_ROW < ROWS && local_COLUMN < COLUMNS){
			for(int k=0; k<COLUMNS; k++){
				tmp += D_A[local_ROW * ROWS + k] * D_B[k * COLUMNS + local_COLUMN];
			}
			D_C[local_index] = tmp;
		}
  }
    // We basically need a wrapper for the actual kernal call since __global__ functions cannot be class members.
    void MM_t::call_gpu_thread_matrix_multiply(X_TYPE* D_A, X_TYPE* D_B, X_TYPE* D_C, int ROWS, int COLUMNS){
        int block_size = 512;
        int grid_size = ((size + block_size) / block_size);
        gpu_thread_matrix_multiply<<<grid_size,block_size>>>(Mesh2D.D_A, Mesh2D.D_B, Mesh2D.D_C, size, size);
    }

    void MM_t::call_cuBLASxgemm(X_TYPE* D_A, X_TYPE* D_B, X_TYPE* D_C, int ROWS, int COLUMNS){
        X_TYPE alpha = 1.2;
        X_TYPE beta = 1.0e-3;
        cublasHandle_t cublasH = NULL;
        /////////Create the cuBLAS Handle
      	CHECK_CUBLAS(cublasCreate(&cublasH));


        ///////////Enableing Automatic Use of Tensor Cores
    	  //CHECK_CUBLAS(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));
        #ifdef USE_DOUBLE
          CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, ROWS, COLUMNS, COLUMNS, &alpha, D_A, COLUMNS, D_B, COLUMNS, &beta, D_C, COLUMNS));
        #else
          CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, ROWS, COLUMNS, COLUMNS, &alpha, D_A, COLUMNS, D_B, COLUMNS, &beta, D_C, COLUMNS));
        #endif
    }


#endif



#ifdef CUDA_ENABLED
__host__ void MM_t::run() 
#else
void MM_t::run() 
#endif
{

//============================================================================================================
//XGEMMS
//============================================================================================================
    if (name == "xgemm" && algorithm == "simple"){
        Initialize_symmetric_matricies_ABC();
        omp_threads = 1;
        do {
            measure();
            simple_matrix_multiply(size, size);
            measure();
            N_runs ++;
        }while (time < max_time && N_runs < max_runs);
        calculate_stats();
        print_info();
    }

    if (name == "xgemm" && algorithm == "openmp")
    {
        Initialize_symmetric_matricies_ABC();
        do {
            measure();
            openmp_matrix_multiply(size, size);
            measure();
            N_runs ++;
        }while (time < max_time && N_runs < max_runs);
      calculate_stats();
      print_info();
    }

    if (name == "xgemm" && algorithm == "cblas")
    {
        Initialize_symmetric_matricies_ABC();
        do {
            measure();
            
            cblas_matrix_multiply(size, size);
            measure();
            N_runs ++;
        }while (time < max_time && N_runs < max_runs);
      calculate_stats();
      print_info();
    }


    if (name == "jacobi" && algorithm == "simple")
    {
        X_TYPE* A = (X_TYPE *) malloc((size * size)*sizeof(X_TYPE));
        X_TYPE* B = (X_TYPE *) malloc((size)*sizeof(X_TYPE));
        X_TYPE* C = (X_TYPE *) malloc((size)*sizeof(X_TYPE));
        X_TYPE* Ctmp = (X_TYPE *) malloc((size)*sizeof(X_TYPE));

        do {
        /* initialize the arrays */
        //initialize_matrix_1D(A, B, C, kernal.size, kernal.size);
        srand(rand());
        for (int row = 0; row < size; row++)
        {
        X_TYPE rowsum = 0.0;
        for (int col = 0; col < size; col++)
        {
          X_TYPE value = rand()/(X_TYPE)RAND_MAX;
          A[row + col*size] = value;
          rowsum += value;
        }
        A[row + row*size] += rowsum;
        B[row] = rand()/(X_TYPE)RAND_MAX;
        C[row] = 0.0;
        Ctmp[row] = 0.0;
      }

            measure();
            simple_jacobi(A, B, C, Ctmp, size, size);
            measure();
            N_runs ++;
        }while (time < max_time && N_runs < max_runs);
      calculate_stats();
      print_info();
    }

if (name == "jacobi" && algorithm == "openmp")
    {
        X_TYPE* A = (X_TYPE *) malloc((size * size)*sizeof(X_TYPE));
        X_TYPE* B = (X_TYPE *) malloc((size)*sizeof(X_TYPE));
        X_TYPE* C = (X_TYPE *) malloc((size)*sizeof(X_TYPE));
        X_TYPE* Ctmp = (X_TYPE *) malloc((size)*sizeof(X_TYPE));

        do {
        /* initialize the arrays */
        //initialize_matrix_1D(A, B, C, kernal.size, kernal.size);
        srand(rand());
        for (int row = 0; row < size; row++)
        {
        X_TYPE rowsum = 0.0;
        for (int col = 0; col < size; col++)
        {
          X_TYPE value = rand()/(X_TYPE)RAND_MAX;
          A[row + col*size] = value;
          rowsum += value;
        }
        A[row + row*size] += rowsum;
        B[row] = rand()/(X_TYPE)RAND_MAX;
        C[row] = 0.0;
        Ctmp[row] = 0.0;
      }

            measure();
            openmp_jacobi(A, B, C, Ctmp, size, size);
            measure();
            N_runs ++;
        }while (time < max_time && N_runs < max_runs);
      calculate_stats();
      print_info();
    }



#ifdef CUDA_ENABLED
    if (name == "xgemm" && algorithm == "gputhread")
    {
        Initialize_symmetric_matricies_ABC();
        cudaGetDevice(&gpuid);  	
  
        int block_size = 512;
        int grid_size = ((size + block_size) / block_size);
        do {
            measure();

            // Transfer data from host to device memory
            cudaMemcpy(Mesh2D.D_A, Mesh2D.A, sizeof(X_TYPE) * (size * size), cudaMemcpyHostToDevice);
            cudaMemcpy(Mesh2D.D_B, Mesh2D.B, sizeof(X_TYPE) * (size * size), cudaMemcpyHostToDevice);

            call_gpu_thread_matrix_multiply(Mesh2D.D_A, Mesh2D.D_B, Mesh2D.D_C, size, size);
            
            // Transfer data from device to host memory
            cudaMemcpy(Mesh2D.C, Mesh2D.D_C, sizeof(X_TYPE) * (size * size), cudaMemcpyDeviceToHost);

            measure();
            N_runs ++;
        }while (time < max_time && N_runs < max_runs);
      calculate_stats();
      print_info();
    }
    if (name == "xgemm" && algorithm == "cublas")
    {
        Initialize_symmetric_matricies_ABC();
        cudaGetDevice(&gpuid);  	
  
        int block_size = 512;
        int grid_size = ((size + block_size) / block_size);
        do {
            measure();

            // Transfer data from host to device memory
            cudaMemcpy(Mesh2D.D_A, Mesh2D.A, sizeof(X_TYPE) * (size * size), cudaMemcpyHostToDevice);
            cudaMemcpy(Mesh2D.D_B, Mesh2D.B, sizeof(X_TYPE) * (size * size), cudaMemcpyHostToDevice);

            call_cuBLASxgemm(Mesh2D.D_A, Mesh2D.D_B, Mesh2D.D_C, size, size);
            
            // Transfer data from device to host memory
            cudaMemcpy(Mesh2D.C, Mesh2D.D_C, sizeof(X_TYPE) * (size * size), cudaMemcpyDeviceToHost);

            measure();
            N_runs ++;
        }while (time < max_time && N_runs < max_runs);
      calculate_stats();
      print_info();
    }

#endif

  
}