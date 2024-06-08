#include "mesh.hpp"

using namespace std;

MM_t::MM_t(Mesh2D_t& Mesh2D_):
  Mesh2D(Mesh2D_)
{
    InitializeMatrix();
    run();
}


MM_t::~MM_t() {
  free(Mesh2D.A);
  free(Mesh2D.B);
  free(Mesh2D.C);
}



// Allocated A, B, C matrixies
void MM_t::InitializeMatrix() 
{
    Mesh2D.size = size;
    
    Mesh2D.A = (X_TYPE *) malloc((Mesh2D.size * Mesh2D.size)*sizeof(X_TYPE));
    Mesh2D.B = (X_TYPE *) malloc((Mesh2D.size * Mesh2D.size)*sizeof(X_TYPE));
    Mesh2D.C = (X_TYPE *) malloc((Mesh2D.size * Mesh2D.size)*sizeof(X_TYPE));

    #ifdef CUDA_ENABLED
        cudaMalloc((void**)&Mesh2D.D_A, sizeof( X_TYPE ) * (Mesh2D.size * Mesh2D.size));
        cudaMalloc((void**)&Mesh2D.D_B, sizeof( X_TYPE ) * (Mesh2D.size * Mesh2D.size));
        cudaMalloc((void**)&Mesh2D.D_C, sizeof( X_TYPE ) * (Mesh2D.size * Mesh2D.size));
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

#ifdef CUDA_ENABLED
    __global__ void gpu_matrix_multiply(X_TYPE* D_A, X_TYPE* D_B, X_TYPE* D_C,int ROWS, int COLUMNS){
    
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
    void MM_t::call_gpu_matrix_multiply(X_TYPE* D_A, X_TYPE* D_B, X_TYPE* D_C, int ROWS, int COLUMNS){
        int block_size = 512;
        int grid_size = ((size + block_size) / block_size);
        gpu_matrix_multiply<<<grid_size,block_size>>>(Mesh2D.D_A, Mesh2D.D_B, Mesh2D.D_C, size, size);
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
        do {
            measure();
            openmp_matrix_multiply(size, size);
            measure();
            N_runs ++;
        }while (time < max_time && N_runs < max_runs);
      calculate_stats();
      print_info();
    }
#ifdef CUDA_ENABLED
    if (name == "xgemm" && algorithm == "simplegpu")
    {
        cudaGetDevice(&gpuid);  	
  
        int block_size = 512;
        int grid_size = ((size + block_size) / block_size);
        do {
            measure();

            // Transfer data from host to device memory
            cudaMemcpy(Mesh2D.D_A, Mesh2D.A, sizeof(X_TYPE) * (size * size), cudaMemcpyHostToDevice);
            cudaMemcpy(Mesh2D.D_B, Mesh2D.B, sizeof(X_TYPE) * (size * size), cudaMemcpyHostToDevice);

            call_gpu_matrix_multiply(Mesh2D.D_A, Mesh2D.D_B, Mesh2D.D_C, size, size);
            
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