#pragma once

#include "Initialize.h"
#include "MM.h"
#include "Mesh.h"

using namespace std;

template<typename T>
class EnerBe {
public:
    void run();
private:
    Initialize2D<T> init;
    Mesh2D<T> mesh;
    MM<T> mm;
};
