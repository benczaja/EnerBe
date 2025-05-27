#include "MM.h"
#include "Profiler.h"
#include <iostream>
#include <omp.h>

// THis is needed because I want to keep the kernal implementations in the cpp
template class MM<double>;
template class MM<float>;
template class MM<_Float16>;


template<typename T>
void MM<T>::naiveGEMM() {
    // Example dummy implementation (matrix multiply)
    std::cout << "[MM] Running naiveGEMM (prec "<<sizeof(T)*8.0<<" bits) on matrices A, B -> C\n";
    std::cout << "[MM] A prec: "<<sizeof(mesh.A[0])*8.0<<"\n"<<endl;
    Profiler profiler;
    
    profiler.timestart = profiler.measureSerialTime();
    for (int i = 0; i < mesh.Nx; ++i) {
        for (int j = 0; j < mesh.Ny; ++j) {
            T sum = static_cast<T>(0.0);
             for (int k = 0; k < mesh.Ny; ++k) {
                sum += mesh.A[i * mesh.Ny + k] * mesh.B[k * mesh.Ny + j];
            }
            mesh.C[i * mesh.Ny + j] = sum; // Just element-wise multiply as a placeholder 
        }
    }
    profiler.timeend = profiler.measureSerialTime();

    std::cout << "[MM] Time taken: " << profiler.timeend - profiler.timestart << " seconds\n";
}


template<typename T>
void MM<T>::tiledGEMM(){
    // Example dummy implementation (matrix multiply)
    std::cout << "[MM] Running tiledGEMM (prec "<<sizeof(T)*8.0<<" bits) on matrices A, B -> C\n";

    int blockSize = 32000/(3* sizeof(T)); // 3200 bytes/ sizeof(T) bytes per element
    Profiler profiler;

    profiler.timestart = profiler.measureSerialTime();
    
    for (int ii =0; ii < mesh.Nx; ii+=blockSize){
        for (int jj =0; jj < mesh.Ny; jj+=blockSize){
            for (int kk =0; kk < mesh.Ny; kk+=blockSize){
                // Loop through the blocks
                for (int i = ii; i < std::min(ii + blockSize, mesh.Nx); ++i) {
                    for (int j = jj; j < std::min(jj + blockSize, mesh.Ny); ++j) {
                        T sum = 0;
                        for (int k = kk; k < std::min(kk + blockSize, mesh.Ny); ++k) {
                            sum += mesh.A[i * mesh.Ny + k] * mesh.B[k * mesh.Ny + j];
                        }
                        mesh.C[i * mesh.Ny + j] += sum; 
                    }
                }
            }
        }
    }
    profiler.timeend = profiler.measureSerialTime();


    std::cout << "[MM] Time taken: " << profiler.timeend - profiler.timestart << " seconds\n";
}

template<typename T>
void MM<T>::naiveOpenMPGEMM() {
    // Example dummy implementation (matrix multiply)
    std::cout << "[MM] Running naiveOpenMPGEMM (prec "<<sizeof(T)*8.0<<" bits) on matrices A, B -> C\n";
    Profiler profiler;
    
    profiler.timestart = profiler.measureOpenMPTime();
    #pragma omp parallel for collapse(2)// Parallelize the outer three
    for (int i = 0; i < mesh.Nx; ++i) {
        for (int j = 0; j < mesh.Ny; ++j) {
            T sum =0;
             for (int k = 0; k < mesh.Ny; ++k) {
                sum += mesh.A[i * mesh.Ny + k] * mesh.B[k * mesh.Ny + j];
            }
            mesh.C[i * mesh.Ny + j] = sum; // Just element-wise multiply as a placeholder 
        }
    }
    profiler.timeend = profiler.measureOpenMPTime();

    std::cout << "[MM] Time taken: " << profiler.timeend - profiler.timestart << " seconds\n";
}


template<typename T>
void MM<T>::tiledOpenMPGEMM(){
    // Example dummy implementation (matrix multiply)
    std::cout << "[MM] Running tiledOpenMPGEMM (prec "<<sizeof(T)*8.0<<" bits) on matrices A, B -> C\n";

    int blockSize = 32000/(3* sizeof(T)); // 3200 bytes/ sizeof(T) bytes per element
    Profiler profiler;

    profiler.timestart = profiler.measureOpenMPTime();
    
    for (int ii =0; ii < mesh.Nx; ii+=blockSize){
        for (int jj =0; jj < mesh.Ny; jj+=blockSize){
            for (int kk =0; kk < mesh.Ny; kk+=blockSize){
                // Loop through the blocks
                #pragma omp parallel for collapse(2)
                for (int i = ii; i < std::min(ii + blockSize, mesh.Nx); ++i) {
                    for (int j = jj; j < std::min(jj + blockSize, mesh.Ny); ++j) {
                        T sum = 0;
                        for (int k = kk; k < std::min(kk + blockSize, mesh.Ny); ++k) {
                            sum += mesh.A[i * mesh.Ny + k] * mesh.B[k * mesh.Ny + j];
                        }
                        mesh.C[i * mesh.Ny + j] += sum; 
                    }
                }
            }
        }
    }
    profiler.timeend = profiler.measureOpenMPTime();
    std::cout << "[MM] Time taken: " << profiler.timeend - profiler.timestart << " seconds\n";
}
