#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cublas_v2.h"
#include "MM.h"
#include "Profiler.h"



// KERNALS
template<typename T>
__global__ void threadCudaGEMM_kernal(T* D_A, T* D_B, T* D_C,int ROWS, int COLUMNS){
    // This will solve the matmul by assigning each GPU thread to a single site on the result matrix C
    // Each GPU thread has a local index in its CUDA thread block
    
    int local_COLUMN = threadIdx.x + blockIdx.x * blockDim.x;
	int local_ROW = threadIdx.y + blockIdx.y * blockDim.y;
	int local_index = local_COLUMN + local_ROW * ROWS; // Right now this only works for symetric matricies
	T tmp = 0;  
    
    if(local_ROW < ROWS && local_COLUMN < COLUMNS){
			for(int k=0; k<COLUMNS; k++){
				tmp += D_A[local_ROW * ROWS + k] * D_B[k * COLUMNS + local_COLUMN];
			}
			D_C[local_index] = tmp;
		}
  }
// Explicit instantiation of the template for different data types
template __global__ void threadCudaGEMM_kernal<__half>(__half*, __half*, __half*, int, int);
template __global__ void threadCudaGEMM_kernal<float>(float*, float*, float*, int, int);
template __global__ void threadCudaGEMM_kernal<double>(double*, double*, double*, int, int);


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

// Helper function (specialized for each type)
template<typename T>
void launch_threadCudaGEMM_kernal(T* D_A, T* D_B, T* D_C, int ROWS, int COLUMNS, int grid_size, int block_size);

template<>
void launch_threadCudaGEMM_kernal<float>(float* D_A, float* D_B, float* D_C, int ROWS, int COLUMNS, int grid_size, int block_size) {
    threadCudaGEMM_kernal<<<grid_size, block_size>>>(D_A, D_B, D_C, ROWS, COLUMNS);
}
template<>
void launch_threadCudaGEMM_kernal<double>(double* D_A, double* D_B, double* D_C, int ROWS, int COLUMNS, int grid_size, int block_size) {
    threadCudaGEMM_kernal<<<grid_size, block_size>>>(D_A, D_B, D_C, ROWS, COLUMNS);
}
template<>
void launch_threadCudaGEMM_kernal<__half>(__half* D_A, __half* D_B, __half* D_C, int ROWS, int COLUMNS, int grid_size, int block_size) {
    threadCudaGEMM_kernal<<<grid_size, block_size>>>(D_A, D_B, D_C, ROWS, COLUMNS);
}




// Host function to call the kernel
template<typename T>
void MM<T>::threadCudaGEMM() {
    int block_size = 512;
    int grid_size = ((mesh.size + block_size) / block_size);
    launch_threadCudaGEMM_kernal<T>(mesh.d_A, mesh.d_A, mesh.d_C, mesh.Nx, mesh.Ny,block_size,grid_size);
}
// Now isntatiate the templates on the host side
template void MM<__half>::threadCudaGEMM();
template void MM<float>::threadCudaGEMM();
template void MM<double>::threadCudaGEMM();