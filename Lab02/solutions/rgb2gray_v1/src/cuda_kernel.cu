// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
// Include C++ header files.
#include <iostream>

// Include associated header file.
#include "../include/cuda_kernel.cuh"


/**
 * Sample CUDA device function which adds an element from array A and array B.
 *
 */
__global__ void rgb2gray(unsigned char * input, unsigned char *output, int rows, int cols){
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  output[i]=(input[i*3+0]+input[i*3+1]+input[i*3+2])/3;
  

}



/**
 * Wrapper function for the CUDA kernel function.
 */
void kernel_rgb2gray(RGBImage *Input, GrayImage *Output) {
    unsigned int n;
    n=Input->height*Input->width;

    unsigned char * d_input;
    unsigned char * d_output;

    cudaMalloc((void **)&d_input, 3*n*sizeof(unsigned char));
    cudaMalloc((void **)&d_output, n*sizeof(unsigned char));

    cudaMemcpy(d_input,Input->data,3*n*sizeof(unsigned char),cudaMemcpyHostToDevice);

    dim3 BlockSize(512,1);
    dim3 GridSize(ceil((n)/BlockSize.x +1), 1);

    rgb2gray<<<GridSize, BlockSize>>>(d_input,d_output, Input->height, Input->width);

    cudaMemcpy(Output->data,d_output,n*sizeof(unsigned char), cudaMemcpyDeviceToHost);
}






