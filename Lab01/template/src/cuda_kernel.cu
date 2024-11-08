// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Include associated header file.
#include "../include/cuda_kernel.cuh"



// TODO: Define the kernel function right here



/**
 * Wrapper function for the CUDA kernel function.
 */
void kernel(float *A, float *B, float *C, int N) {
    // TODO: create the device pointers

    // TODO: allocate device memory. Hint: used cudaMalloc

    // TODO: Copy data from host to device. Hint: use cudaMemcpy

    // TODO: define the thread dimentions 
    dim3 blockSize;
    dim3 gridSize;
    // TODO: Issue the kernel on the GPU 
    VectorAdd<<<gridSize, blockSize>>>(??, ??, ??);
    
    // TODO: Copy the computed results from device to host
}











