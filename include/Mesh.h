#pragma once
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif


using namespace std;

// A Class to represent directed graph
template<typename T>
class Mesh2D {

public:
    int size;
    int Nx; // No. of vertices
    int Ny; // No. of vertices

    T* A;
    T* B;
    T* C;

    #ifdef ENABLE_CUDA
        // Device pointers for CUDA
        T* d_A;
        T* d_B;
        T* d_C;
    #endif


    Mesh2D(int Nx_, int Ny_)
        : Nx(Nx_), Ny(Ny_), A(nullptr), B(nullptr), C(nullptr)
    {
        size = Nx * Ny;
    }

    ~Mesh2D() {
        delete[] A;
        delete[] B;
        delete[] C;
        #ifdef ENABLE_CUDA
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
        #endif
    }

};