// Include C++ header files.
#include <iostream>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"

#define N 1024


int main(void) {

    float A[N];
    float B[N];
    float C[N];

    for(int i=0;i<N;i++){
        A[i]=rand();
        B[i]=rand();
    }

    kernel(A,B,C,N);
    
    for(int i=0;i<N;i++){
        std::cout << A[i] << " + " << B[i]<< " = " << C[i] << std::endl;
    }

    printf("\nProgramm Finished!\n");
    return 0;
}