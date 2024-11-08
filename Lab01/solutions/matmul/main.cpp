// Include C++ header files.
#include <iostream>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"

#define M 1024
#define N 1024
#define K 1024


int main(void) {

    float *A;
    float *B;
    float *C;

    A = (float *)malloc(M*K*sizeof(float));
    B = (float *)malloc(K*N*sizeof(float));
    C = (float *)malloc(M*N*sizeof(float));


    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            A[N*i+j]=rand();
            B[N*i+j]=rand();
        }
    }

    kernel(A,B,C,M,N,K);

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
           std::cout << C[i*N+j] << std::endl;
        }
    }

    printf("\nProgramm Finished!\n");
    return 0;
}