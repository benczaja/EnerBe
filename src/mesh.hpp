#include <omp.h> // needed for OpenMP 
#include <math.h>
#include "EnerBe.h"

using namespace std;

// A Class to represent directed graph
class Mesh2D_t {

public:
    int size;
    int Nx; // No. of vertices
    int Ny; // No. of vertices

    X_TYPE* A;
    X_TYPE* B;
    X_TYPE* C;

    #ifdef CUDA_ENABLED
        X_TYPE* D_A;
        X_TYPE* D_B;
        X_TYPE* D_C;
    #endif

};

class MM_t : public EnerBe {
    
    private:
    
        Mesh2D_t& Mesh2D;

    public:

        MM_t(Mesh2D_t& Mesh2D_);
        ~MM_t();

        void InitializeMatrix();
        void PrintElement(int i);
        void simple_matrix_multiply(int ROWS, int COLUMNS);
        void openmp_matrix_multiply(int ROWS, int COLUMNS);
        #ifdef CUDA_ENABLED
            void call_gpu_matrix_multiply(X_TYPE* D_A, X_TYPE* D_B, X_TYPE* D_C, int ROWS, int COLUMNS);
            void run();
        #else
            void run();
        #endif
};


