// Include C++ header files.
#include <iostream>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"

#define N 1024

time_t start, end; 

int main(void) {

    // TODO: Create empty arrays on the host (CPU)

    // TODO: initialize the arrays with random data (e.g., use rand function)
    time(&start); 
    // TODO: Peform the computation on the CPU 
    time(&end); 
    double time_taken = double(end - start); 
    // TODO: call a function passing the pointer of the arrays as arguments to compute on the GPU
    
    // TODO: verify the results.

    printf("\nProgramm Finished!\n");
    return 0;
}