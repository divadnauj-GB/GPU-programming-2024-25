// Include C++ header files.
#include <iostream>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"
#include <math.h>

#define M 500
#define N 1000
#define K 300



void cpu_MatMul(float *A, float *B, float *C, int m, int n, int k){
    for(int row=0; row<m; row++){
        for (int col=0; col<n; col++){
            for (int ii = 0; ii < k; ii++) {
                C[row * n + col] += A[row * k + ii] * B[ii * n + col];
            }
        }
    }
}


int main(void) {

    float *A;
    float *B;
    float *C, *C_cpu;

    A = (float *)malloc(M*K*sizeof(float));
    B = (float *)malloc(K*N*sizeof(float));
    C = (float *)malloc(M*N*sizeof(float));

    C_cpu = (float *)malloc(M*N*sizeof(float));

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            A[K*i+j]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            B[N*i+j]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
    }

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            C[N*i+j]=0;
            C_cpu[N*i+j]=0;
        }
    }

    cpu_MatMul(A,B,C_cpu,M,N,K);

    kernel(A,B,C,M,N,K);
    bool error = false;
    float diff = 0.0;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            diff = abs(C[i*N+j] - C_cpu[i*N+j]);
            if (diff>10e-4){            
                error = true;
                std::cout << i << " " << j << " " << diff << " " << C[i*N+j] << " " << C_cpu[i*N+j] << std::endl;
           }           
        }
        //std::cout << std::endl;
    }

    if (error==true){
       printf("\nThe Results are Different!\n");
    }else{
        printf("\nThe Results match!\n");
    }

    printf("\nProgramm Finished!\n");
    return 0;
}