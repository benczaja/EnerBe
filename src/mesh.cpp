#include "mesh.hpp"

using namespace std;

MM_t::MM_t(Mesh2D_t& Mesh2D_):
  Mesh2D(Mesh2D_)
{
    InitializeMatrix();
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
