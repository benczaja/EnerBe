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

    #if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
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

        void Initialize_symmetric_matricies_ABC();
        void PrintElement(int i);
        void simple_matrix_multiply(int ROWS, int COLUMNS);
        void openmp_matrix_multiply(int ROWS, int COLUMNS);
        void cblas_matrix_multiply(int ROWS, int COLUMNS);
        void simple_jacobi(X_TYPE* A, X_TYPE* B, X_TYPE* C, X_TYPE* Ctmp, int ROWS, int COLUMNS);
        void openmp_jacobi(X_TYPE* A, X_TYPE* B, X_TYPE* C, X_TYPE* Ctmp, int ROWS, int COLUMNS);
        #if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
            void call_gpu_thread_matrix_multiply(X_TYPE* D_A, X_TYPE* D_B, X_TYPE* D_C, int ROWS, int COLUMNS);
            void call_gpu_BLASxgemm(X_TYPE* D_A, X_TYPE* D_B, X_TYPE* D_C, int ROWS, int COLUMNS);
            void run();
        #else
            void run();
        #endif
};


