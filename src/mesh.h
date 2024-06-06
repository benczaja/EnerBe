#include <ctype.h> // needed for isdigit
#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘EXIT_FAILURE’ 
#include <string.h> // needed for strcmp
#include <iostream> // needed for CPP IO ... cout, endl etc etc
#include <vector>
#include <bits/stdc++.h> 
#include <stdbool.h> // needed for bool usage
#include <omp.h> // needed for OpenMP 
#include <math.h>
#include "EnerBe.h"

using namespace std;

// A Class to represent directed graph
class Mesh2D {

private:
    int Nx; // No. of vertices
    int Ny; // No. of vertices

 
public:
    Mesh2D(int N); // Constructor
    // function to add an edge to graph
    void print();
    void initialize_matrix();
};
 
Mesh2D::Mesh2D(int N)
{
    this->N = N;
    cout << "Object is being created" << endl;
    //X_TYPE* A = (X_TYPE*)malloc((N * N) * sizeof(X_TYPE));
    // Create a dynamic array of pointers
    //A = new bool*[N];
    X_TYPE* A = new X_TYPE[N * N];
    
    //// Create a row for every pointer
    //for (int i = 0; i < (N*N); i++) {
    //    // Note : Rows may not be contiguous
    //    A[i] = new X_TYPE[N*N];
 //
    //    // Initialize all entries as false to indicate
    //    // that there are no edges initially
    //    memset(A[i], 0.0, N * sizeof(X_TYPE));
    //}
}
 
// Utility method to print adjacency matrix
void Mesh2D::print()
{
    for (int i = 0; i < (N); i++) {
            cout << A[i] << " ";
        cout << endl;
    }
}


void Mesh2D::initialize_matrix()
{
    unsigned int globalSeed = clock();  
    cout<<"Size: "<<N<<endl;

    for (int i = 0; i < (N*N); i++)
        {
        cout<<i<<endl;
        unsigned int randomState = i ^ globalSeed;
        A[i] = 2.0;
        //A[i] = (X_TYPE) rand_r(&randomState) / RAND_MAX;
        //B[i] = (X_TYPE) rand_r(&randomState) / RAND_MAX;
        }
    cout<<"finished with A"<<endl;
}




class MM : public EnerBe {

    public:

    //X_TYPE* A = new X_TYPE[size * size];
    //X_TYPE* B = new X_TYPE[size * size];
    //X_TYPE* C = new X_TYPE[size * size];
    //X_TYPE* A = (X_TYPE*)malloc((size * size) * sizeof(X_TYPE));
    //X_TYPE* B = (X_TYPE*)malloc((size * size) * sizeof(X_TYPE));
    //X_TYPE* C = (X_TYPE*)malloc((size * size) * sizeof(X_TYPE));

    //std::vector<int> V(10, 0);
    //vector<string> name = vector<string>(5);
    std::vector<X_TYPE> A = std::vector<X_TYPE>(size*size);
    std::vector<X_TYPE> B = std::vector<X_TYPE>(size*size);
    std::vector<X_TYPE> C = std::vector<X_TYPE>(size*size);

    //std::vector <X_TYPE> A(10, 0); 
    //std::vector <X_TYPE> B(10, 0); 
    //std::vector <X_TYPE> C(10, 0); 

 
    void initialize_matrix_1D(){
            std::cout<<"Inititializing MM"<<std::endl;


            //generate(A.begin(), A.end(), (X_rand); 
            //generate(B.begin(), B.end(), (X_TYPE) rand); 
            //generate(C.begin(), C.end(), (X_TYPE) rand); 

            //generate(A.begin(), B.end(), rand); 
            //generate(A.begin(), B.end(), rand); 
            unsigned int globalSeed = clock();  
            std::cout<<"Random number"<<std::endl;
            std::cout<<size<<std::endl;
            std::cout<<A.size()<<std::endl;

            for (size_t i = 0; i < A.size(); i++)
                {
                std::cout<<i<<std::endl;
                 unsigned int randomState = i ^ globalSeed;
                A[i] = (X_TYPE) rand_r(&randomState) / RAND_MAX;
                B[i] = (X_TYPE) rand_r(&randomState) / RAND_MAX;
                }

               for (int i = 0; i < size; i++) { 
                    std::cout << A[i] << " "; 
                } 

            std::cout<<"finished with A"<<std::endl;

            // Do this in Parallel with OpenMP
            // Needs a seperate seed per thread as rand() is obtaining a mutex and therefore locking each thread.
            //std::cout<<size<<std::endl;
            //std::cout<<sizeof(A)<<std::endl;
            //std::cout<<std::begin(A)<<std::endl;
            //std::cout<<sizeof(A[0])<<std::endl;
            //unsigned int globalSeed = clock();  
            //#pragma omp parallel for
            //for (int i = 0; i < int(size * size); i++)
            //    {
            //    std::cout<<i<<std::endl;
             //   unsigned int randomState = i ^ globalSeed;
            //    A[i] = (X_TYPE) rand_r(&randomState) / RAND_MAX;
            //    B[i] = (X_TYPE) rand_r(&randomState) / RAND_MAX;
            //    C[i] = 0.0 ;
            //}
        }

        void initialize_matrix_2D(X_TYPE** A, X_TYPE** B, X_TYPE** C, int ROWS, int COLUMNS){
            for (int i = 0 ; i < ROWS ; i++)
            {
                for (int j = 0 ; j < COLUMNS ; j++)
                {
                    A[i][j] = (X_TYPE) rand() / RAND_MAX ;
                    B[i][j] = (X_TYPE) rand() / RAND_MAX ;
                    C[i][j] = 0.0 ;
                }
            }
        }

//============================================================================================================
//XGEMMS
//============================================================================================================
void simple_matrix_multiply(int ROWS, int COLUMNS){//X_TYPE* A, X_TYPE* B, X_TYPE* C, int ROWS, int COLUMNS){
    
    for(int i=0;i<ROWS;i++)
    {
        for(int j=0;j<COLUMNS;j++)
        {
            for(int k=0;k<COLUMNS;k++)
            {
                // ROWS and COLS might be messed up here
                C[i * COLUMNS + j] += A[i * ROWS + k] * B[k * COLUMNS + j];
            }
        }
    }
}

void openmp_matrix_multiply(int ROWS, int COLUMNS){
    
    #pragma omp parallel for 
    for (int i = 0; i < ROWS; ++i) 
    {
        for (int j = 0; j < COLUMNS; ++j) 
        {
            for (int k = 0; k < COLUMNS; ++k) 
            {
                C[i * COLUMNS + j] += A[i * ROWS + k] * B[k * COLUMNS + j];
            }
        }
    }
}

};


