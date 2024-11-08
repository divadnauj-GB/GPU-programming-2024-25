// Include C++ header files.
#include <iostream>
#include<math.h>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"

#define N 10000



void cpu_saxpy_vect(float *A, float *B, float *C, int n){
    for (int i=0; i<n ; i++){
        C[i] += A[i]*B[i];
    }
}

int main(void) {

    float A[N];
    float B[N];
    float C[N], C_cpu[N];

    for(int i=0;i<N;i++){
        A[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        B[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        C_cpu[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        C[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    cpu_saxpy_vect(A,B,C_cpu,N);

    kernel(A,B,C,N);
    
    bool error = false;
    float diff = 0.0;
    for(int i=0;i<N;i++){
        diff = abs(C[i] - C_cpu[i]);
        if (diff>10e-4){            
            error = true;
            std::cout << i << " " << diff << " " << C[i] << " " << C_cpu[i] << std::endl;
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