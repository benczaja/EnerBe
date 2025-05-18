#pragma once
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


    Mesh2D(int Nx_, int Ny_)
        : Nx(Nx_), Ny(Ny_), A(nullptr), B(nullptr), C(nullptr)
    {
        size = Nx * Ny;
    }

    ~Mesh2D() {
        delete[] A;
        delete[] B;
        delete[] C;
    }

};