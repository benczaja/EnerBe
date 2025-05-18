#include "Initialize.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace std;


// Allocated A, B, C matrixies
template<typename T>
void Initialize2D<T>::initializeMatrices()
{
    cout<<"Initializing 2D Matricies of size ("<<mesh.Nx<<","<<mesh.Ny<<")"<<endl;
    mesh.A = (T *) malloc((mesh.Nx * mesh.Ny)*sizeof(T));
    mesh.B = (T *) malloc((mesh.Nx * mesh.Ny)*sizeof(T));
    mesh.C = (T *) malloc((mesh.Nx * mesh.Ny)*sizeof(T));

    unsigned int globalSeed = clock();  

    for (int i = 0; i < (mesh.size); i++)
    {
        unsigned int randomState = i ^ globalSeed;
        mesh.A[i] = (T) rand_r(&randomState) / RAND_MAX;
        mesh.B[i] = (T) rand_r(&randomState) / RAND_MAX;
        mesh.C[i] = 0.0;
    }

}
