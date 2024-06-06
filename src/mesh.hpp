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

};

class MM_t : public EnerBe {
    
    private:
    
        Mesh2D_t& Mesh2D;

    public:

        MM_t(Mesh2D_t& Mesh2D_);
        ~MM_t();

        void InitializeMatrix();
        void PrintElement(int i);
        void run();
        void simple_matrix_multiply(int ROWS, int COLUMNS);
        void openmp_matrix_multiply(int ROWS, int COLUMNS);
};


