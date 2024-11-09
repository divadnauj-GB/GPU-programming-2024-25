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
__global__ void kernel_1dconv(float *A, float *B, float *C){
    __shared__ float temp[BLOCK_SIZE + 2 * RADIUS];
    __shared__ float temp2[2 * RADIUS+1];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    temp[lindex] = A[gindex];
    if (threadIdx.x < RADIUS) {
      temp[lindex - RADIUS] = A[gindex - RADIUS];
      temp[lindex + BLOCK_SIZE] = A[gindex + BLOCK_SIZE];
    }
    
    if (threadIdx.x <= 2*RADIUS) {
        temp2[threadIdx.x]=B[threadIdx.x];
    }
    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    float result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
      result += temp[lindex + offset]*temp2[offset+RADIUS];

    // Store the result
    C[gindex] = result;
}



/**
 * Wrapper function for the CUDA kernel function.
 */
void kernel(float *A, float *B, float *C, int Nn) {
    // Launch CUDA kernel.
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, Nn*sizeof(float));
    cudaMalloc((void**) &d_B, (2*RADIUS+1)*sizeof(float));
    cudaMalloc((void**) &d_C, Nn*sizeof(float));

    cudaMemcpy(d_A, A, Nn*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (2*RADIUS+1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, Nn*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE,1,1);
    dim3 gridSize(Nn/BLOCK_SIZE+1,1,1);

    kernel_1dconv<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    
    cudaMemcpy(C, d_C, Nn*sizeof(float), cudaMemcpyDeviceToHost);
}











