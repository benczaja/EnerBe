#include "Mesh.h"
#include <cstdlib>
#include <cstddef>

template<typename T>
Mesh2D::Mesh2D(int Nx_, int Ny_) : Nx(Nx_), Ny(Ny_), A(nullptr), B(nullptr), C(nullptr)
{
    size = Nx * Ny;
}
template<typename T>
Mesh2D::~Mesh2D() {
    delete[] A;
    delete[] B;
    delete[] C;
}
