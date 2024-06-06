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
    
    Mesh2D.A = (X_TYPE *) malloc((Mesh2D.size *Mesh2D.size)*sizeof(X_TYPE));
    Mesh2D.B = (X_TYPE *) malloc((Mesh2D.size *Mesh2D.size)*sizeof(X_TYPE));
    Mesh2D.C = (X_TYPE *) malloc((Mesh2D.size *Mesh2D.size)*sizeof(X_TYPE));

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

void MM_t::run() 
{

//============================================================================================================
//XGEMMS
//============================================================================================================
    if (name == "xgemm" && algorithm == "simple"){
        
        clock_t t; // declare clock_t (long type)
        omp_threads = 1;
        
        do {
            start = double(clock());
            simple_matrix_multiply(size, size);
            end = double(clock());
            times[N_runs] =  (end - start)/CLOCKS_PER_SEC;
            N_runs ++;
        }while (time < max_time && N_runs < max_runs);
    
        calculate_stats();
        print_info();

    }

  if (name == "xgemm" && algorithm == "openmp")
  {
    // omp_get_wtime needed here because clock will sum up time for all threads
    do {
      start = omp_get_wtime();  
      openmp_matrix_multiply(size, size);
      end = omp_get_wtime(); 
      times[N_runs] += end - start;
      N_runs ++;
    }while (time < max_time && N_runs < max_runs);
    calculate_stats();
    print_info();
  }




  
}