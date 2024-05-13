
__global__ void gpu_axpy(int n, X_TYPE a, X_TYPE * x, X_TYPE * y) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handling arbitrary vector size
    if (tid < n){
        y[tid] = a * x[tid] + y[tid];
    }
}


__global__ void simple_matrix_multiply(X_TYPE* D_A, X_TYPE* D_B, X_TYPE* D_C, int ROWS, int COLUMNS){
    
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