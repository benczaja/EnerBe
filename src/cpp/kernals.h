#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘RAND_MAX’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc

//============================================================================================================
// AXPYS
//============================================================================================================
void simple_axpy(int n, X_TYPE a, X_TYPE * x, X_TYPE * y){

    for(int i=0; i<n; i++){
        y[i] = a * x[i] + y[i];
    }
}

void openmp_axpy(int n, X_TYPE a, X_TYPE * x, X_TYPE * y){

    #pragma omp parallel for
    for(int i=0; i<n; i++){
        y[i] = a * x[i] + y[i];
    }
}


//============================================================================================================
//XGEMMS
//============================================================================================================
void simple_matrix_multiply(X_TYPE** A, X_TYPE** B, X_TYPE** C, int ROWS, int COLUMNS){
    
    for(int i=0;i<ROWS;i++)
    {
        for(int j=0;j<COLUMNS;j++)
        {
            for(int k=0;k<COLUMNS;k++)
            {
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

void openmp_matrix_multiply(X_TYPE** A, X_TYPE** B, X_TYPE** C, int ROWS, int COLUMNS){
    
    #pragma omp parallel for 
    for (int i = 0; i < ROWS; ++i) 
    {
        for (int j = 0; j < COLUMNS; ++j) 
        {
            for (int k = 0; k < COLUMNS; ++k) 
            {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }
}


