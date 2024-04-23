#include <stdio.h> // needed for ‘printf’ 
#include <omp.h> // needed for OpenMP 
#include <time.h> // needed for clock() and CLOCKS_PER_SEC etc
#include "../helper.h" // local helper header to clean up code
#include "kernals.h"

int main( int argc, char *argv[] )  {

    kernal kernal;
    kernal.name = "axpy";

    /* VERY DUMB Argument Parsers */
    kernal.size = parse_arguments(argc, argv, &simple, &openmp, &sanity_check);

    X_TYPE *sx; /* n is an array of N integers */
    X_TYPE *sy; /* n is an array of N integers */

    sx = (X_TYPE*)malloc(kernal.size * sizeof (X_TYPE));
    sy = (X_TYPE*)malloc(kernal.size * sizeof (X_TYPE));

    /* Simple saxpy */
    /*==============================*/
    if (true == simple)
    {
        kernal.algorithm = "simple";
        clock_t t; // declare clock_t (long type)
        t = clock(); // start the clock
    
        simple_axpy(kernal.size, 2.0, sx, sy);
    
        t = clock() - t; // stop the clock    
        kernal.time = ((double)t)/CLOCKS_PER_SEC; // convert to seconds (and long to double)
    }

    /* OpenMP parallel saxpy */
    /*==============================*/
    if (true == openmp)
    {
        kernal.algorithm = "openmp";
        // omp_get_wtime needed here because clock will sum up time for all threads
        double start = omp_get_wtime();  

        openmp_axpy(kernal.size, 2.0, sx, sy);
    
        double end = omp_get_wtime();

        kernal.time = end-start;
    }
    kernal.print_info();
    free(sx);
    free(sy);
}
