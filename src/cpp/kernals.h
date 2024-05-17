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

//============================================================================================================
//Jacbobi Iterative solver
//============================================================================================================
void simple_jacobi(X_TYPE* A, X_TYPE* B, X_TYPE* C, X_TYPE* Ctmp, int ROWS, int COLUMNS)
//int simple_jacobi(double *A, double *b, double *x, double *xtmp)
{
// Returns the number of iterations performed

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
        //printf("row: %d\n",row);
      dot = 0.0;
      for (col = 0; col < COLUMNS; col++)
      {
        //printf("col: %d\n",col);

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

