#include <stdio.h> // needed for ‘printf’ 
#include <stdlib.h> // needed for ‘RAND_MAX’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "mesh.hpp"
#include "argparser.h"
//#include "kernals.h"

using namespace std;

int main( int argc, char *argv[] )  {
    
  Mesh2D_t Mesh2D;
  MM_t MM(Mesh2D);

  parse_arguments(argc, argv, MM.size, MM.algorithm, MM.name);

  MM.InitializeMatrix();

  MM.run();

}