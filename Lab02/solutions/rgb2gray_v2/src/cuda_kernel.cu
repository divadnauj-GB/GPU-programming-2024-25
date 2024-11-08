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
__global__ void rgb2gray(pixel * input, unsigned char *output, int rows, int cols){
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  output[i]=(input[i].r+input[i].g+input[i].b)/3;
  

}



/**
 * Wrapper function for the CUDA kernel function.
 */
void kernel_rgb2gray(RGBImage *Input, GrayImage *Output) {
    unsigned int n;
    n=Input->height*Input->width;

    pixel* n_input;
    pixel* d_input;
    unsigned char * d_output;

    n_input = (pixel *)malloc(n*sizeof(pixel));

    cudaMalloc((void **)&d_input, n*sizeof(pixel));
    cudaMalloc((void **)&d_output, n*sizeof(unsigned char));

    for(int i=0; i<n; i++){
      n_input[i].r = Input->data[i*3+0];
      n_input[i].g = Input->data[i*3+1];
      n_input[i].b = Input->data[i*3+2];
    }

    cudaMemcpy(d_input,n_input,n*sizeof(pixel),cudaMemcpyHostToDevice);

    dim3 BlockSize(512,1);
    dim3 GridSize(ceil((n)/BlockSize.x +1), 1);

    rgb2gray<<<GridSize, BlockSize>>>(d_input,d_output, Input->height, Input->width);

    cudaMemcpy(Output->data,d_output,n*sizeof(unsigned char), cudaMemcpyDeviceToHost);
}






