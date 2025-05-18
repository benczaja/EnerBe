#pragma once
#include "Mesh.h"

template<typename T>
class MM {
private:
    Mesh2D<T>& mesh;

public:
    MM(Mesh2D<T>& mesh_) : mesh(mesh_) {}

    ~MM() {}

    void naiveGEMM(); 
    void tiledGEMM(); 
};
