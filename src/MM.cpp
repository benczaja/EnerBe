#include "MM.h"
#include "Profiler.h"
#include <iostream>

// THis is needed because I want to keep the kernal implementations in the cpp
template class MM<double>;
template class MM<float>;




template<typename T>
void MM<T>::naiveGEMM() {
    // Example dummy implementation (matrix multiply)
    std::cout << "[MM] Running naiveGEMM (prec "<<sizeof(T)*8.0<<" bits) on matrices A, B -> C\n";
    Profiler profiler;
    
    profiler.timestart = profiler.measureTime();
    int N = mesh.Nx * mesh.Ny;
    for (int i = 0; i < N; ++i) {
        mesh.C[i] = mesh.A[i] * mesh.B[i]; // Just element-wise multiply as a placeholder
    }
    profiler.timeend = profiler.measureTime();

    std::cout << "[MM] Time taken: " << profiler.timeend - profiler.timestart << " seconds\n";
}
