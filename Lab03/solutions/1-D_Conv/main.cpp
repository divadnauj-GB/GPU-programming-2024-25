// Include C++ header files.
#include <iostream>
#include<math.h>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"


void cpu_1dconv_vect(float *A, float *B, float *C, int n){
    for (int i=1; i<n-1 ; i++){
        for(int offset=-RADIUS;offset<=RADIUS;offset++){
            C[i] += A[i+offset]*B[RADIUS+offset];
        }        
    }
    C[0]=A[0];
    C[n-1]=A[n-1];
}

int main(void) {

    float A[N];
    float Kernel[2*RADIUS+1];

    float C[N], C_cpu[N];

    for(int i=0;i<N;i++){
        A[i]=1;//static_cast <float> (rand()) / static_cast <float> (RAND_MAX);        
        C_cpu[i]=0;
        C[i]=0;
    }

    for(int i=0;i<=2*RADIUS;i++){
        Kernel[i]=0.5; 
    }

    cpu_1dconv_vect(A,Kernel,C_cpu,N);

    kernel(A,Kernel,C,N);
    
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