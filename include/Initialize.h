#pragma once

#include <map>
#include <string>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "Mesh.h"

template<typename T>
class Initialize2D {
private:
    Mesh2D<T>& mesh;

public:
    // Constructor
    Initialize2D(Mesh2D<T>& mesh_) : mesh(mesh_) {}

    // Destructor
    ~Initialize2D() {}

    // Fills A, B, C
    void initializeMatrices() {
        std::cout << "[Init] Initializing 2D Matricies of size (" << mesh.Nx << "," << mesh.Ny << ")" << std::endl;
        mesh.A = (T *) malloc((mesh.Nx * mesh.Ny) * sizeof(T));
        mesh.B = (T *) malloc((mesh.Nx * mesh.Ny) * sizeof(T));
        mesh.C = (T *) malloc((mesh.Nx * mesh.Ny) * sizeof(T));

        unsigned int globalSeed = std::clock();

        for (int i = 0; i < (mesh.Nx * mesh.Ny); i++) {
            unsigned int randomState = i ^ globalSeed;
            mesh.A[i] = (T) rand_r(&randomState) / RAND_MAX;
            mesh.B[i] = (T) rand_r(&randomState) / RAND_MAX;
            mesh.C[i] = static_cast<T>(0.0);
        }
    }
};

