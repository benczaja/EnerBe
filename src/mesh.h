#include <ctype.h> // needed for isdigit
#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘EXIT_FAILURE’ 
#include <string.h> // needed for strcmp
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include <stdbool.h> // needed for bool usage
#include <omp.h> // needed for OpenMP 
#include <math.h>
#include "EnerBe.h"

class mesh2d : public EnerBe {

    public:

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
};


