// Include C++ header files.
# include <stdio.h>
#include <iostream>
#include<math.h>

// Include local CUDA header files.
#include "include/cuda_kernel.cuh"


void cpu_2dconv_vect(float *A, float *B, float *C, int n){
    float x_tmp=0;
    for (int i=0; i<n ; i++){
        for(int j=0; j<n ; j++){
            for(int row=-RADIUS;row<=RADIUS;row++){
                for(int col=-RADIUS;col<=RADIUS;col++){
                    if((i+row)<0 || (j+col)<0){
                        x_tmp=0;
                    }else if((i+row)>=n || (j+col)>=n){
                        x_tmp=0;
                    }else{
                        x_tmp=A[(i+row)*n+(j+col)];
                    }
                    C[i*n+j] += x_tmp*B[(row+RADIUS)*(2*RADIUS+1)+(col+RADIUS)];
                }                
            }    
        }    
    }

}

int main(void) {
    float *A, *Kernel, *C, *C_cpu;

    A = new float[DSIZE*DSIZE];
    Kernel= new float [(2*RADIUS+1)*(2*RADIUS+1)];

    C= new float[DSIZE*DSIZE];
    C_cpu=new float[DSIZE*DSIZE];

    for(int i=0;i<DSIZE*DSIZE;i++){
        A[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);        
        C_cpu[i]=0;
        C[i]=0;
    }

    for(int i=0;i<(2*RADIUS+1)*(2*RADIUS+1);i++){
        Kernel[i]=static_cast <float> (rand()) / static_cast <float> (RAND_MAX);          
    }

    cpu_2dconv_vect(A,Kernel,C_cpu,DSIZE);

    kernel(A,Kernel,C,DSIZE);
    
    bool error = false;
    float diff = 0.0;
    for(int i=0;i<DSIZE*DSIZE;i++){
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