#pragma once
#include "Mesh.h"

template<typename T>
class MM {
private:
    Mesh2D<T>& mesh;

public:
    MM(Mesh2D<T>& mesh_) : mesh(mesh_) {}

    ~MM() {}

    // Serial Implementations
    void naiveGEMM(); 
    void tiledGEMM(); 
    // OpenMP Implementations
    void naiveOpenMPGEMM();
    void tiledOpenMPGEMM();
    // CUDA Implementations
    void threadCudaGEMM();
    void threadCudaGEMM_kernal();
};
