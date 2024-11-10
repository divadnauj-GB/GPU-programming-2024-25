// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

// Include associated header file.
#include "../include/cuda_kernel.cuh"


/**
 * Sample CUDA device function which adds an element from array A and array B.
 *
 */
__global__ void VectorAdd(float *A, float *B, float *C){
   int tid = blockDim.x*blockIdx.x + threadIdx.x;
   C[tid] += A[tid]*B[tid];
}



/**
 * Wrapper function for the CUDA kernel function.
 */
void kernel(float *A, float *B, float *C, int N) {
    // Launch CUDA kernel.
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, N*sizeof(float));
    cudaMalloc((void**) &d_B, N*sizeof(float));
    cudaMalloc((void**) &d_C, N*sizeof(float));

    cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(512,1,1);
    dim3 gridSize(N/512+1,1,1);

    VectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    
    cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
}











