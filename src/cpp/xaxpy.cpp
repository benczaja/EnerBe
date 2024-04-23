#include <stdio.h> // needed for ‘printf’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
#include "kernals.h"

int main( int argc, char *argv[] )  {

    /* VERY DUMB Argument Parsers */
    int N = parse_arguments(argc, argv, &simple, &openmp, &sanity_check);

    X_TYPE *sx; /* n is an array of N integers */
    X_TYPE *sy; /* n is an array of N integers */

    sx = (X_TYPE*)malloc(N * sizeof (X_TYPE));
    sy = (X_TYPE*)malloc(N * sizeof (X_TYPE));

    /* Simple saxpy */
    /*==============================*/
    if (true == simple)
    {
        clock_t t; // declare clock_t (long type)
        t = clock(); // start the clock
    
        simple_axpy(N, 2.0, sx, sy);
    
        t = clock() - t; // stop the clock    
        double time_taken = ((double)t)/CLOCKS_PER_SEC; // convert to seconds (and long to double)
        
        printf("CLASS: axpy\n");
        printf("ALGO: simple\n");
        printf("PRECISION: %d bytes \n",sizeof (X_TYPE));
        printf("SIZE: %d \n",N);
        printf("TIME: %f s\n",time_taken);
    }

    /* OpenMP parallel saxpy */
    /*==============================*/
    if (true == openmp)
    {

        // omp_get_wtime needed here because clock will sum up time for all threads
        double start = omp_get_wtime();  

        openmp_axpy(N, 2.0, sx, sy);
    
        double end = omp_get_wtime();
        printf("CLASS: axpy\n");
        printf("ALGO: openmp\n");
        printf("PRECISION: %d bytes \n",sizeof (X_TYPE));
        printf("OMP_THREADS: %d\n",omp_get_max_threads());
        printf("SIZE: %d \n",N);
        printf("TIME: %f s\n",(end-start));

    }


}
