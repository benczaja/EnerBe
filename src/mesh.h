#include <ctype.h> // needed for isdigit
#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘EXIT_FAILURE’ 
#include <string.h> // needed for strcmp
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include <stdbool.h> // needed for bool usage
#include <omp.h> // needed for OpenMP 
#include <math.h>
#include "EnerBe.h"

class MM : public EnerBe {

    public:

    X_TYPE* A = (X_TYPE*)malloc((size * size) * sizeof(X_TYPE));
    X_TYPE* B = (X_TYPE*)malloc((size * size) * sizeof(X_TYPE));
    X_TYPE* C = (X_TYPE*)malloc((size * size) * sizeof(X_TYPE));

 
    void initialize_matrix_1D(){
            // Do this in Parallel with OpenMP
            // Needs a seperate seed per thread as rand() is obtaining a mutex and therefore locking each thread.
            std::cout<<std::endl;
            unsigned int globalSeed = clock();  
            //#pragma omp parallel for
            for (int i = 0; i < size * size; i++)
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

//============================================================================================================
//XGEMMS
//============================================================================================================
void simple_matrix_multiply(X_TYPE* A, X_TYPE* B, X_TYPE* C, int ROWS, int COLUMNS){
    
    for(int i=0;i<ROWS;i++)
    {
        for(int j=0;j<COLUMNS;j++)
        {
            for(int k=0;k<COLUMNS;k++)
            {
                // ROWS and COLS might be messed up here
                C[i * COLUMNS + j] += A[i * ROWS + k] * B[k * COLUMNS + j];
            }
        }
    }
}

void openmp_matrix_multiply(X_TYPE* A, X_TYPE* B, X_TYPE* C, int ROWS, int COLUMNS){
    
    #pragma omp parallel for 
    for (int i = 0; i < ROWS; ++i) 
    {
        for (int j = 0; j < COLUMNS; ++j) 
        {
            for (int k = 0; k < COLUMNS; ++k) 
            {
                C[i * COLUMNS + j] += A[i * ROWS + k] * B[k * COLUMNS + j];
            }
        }
    }
}

};


