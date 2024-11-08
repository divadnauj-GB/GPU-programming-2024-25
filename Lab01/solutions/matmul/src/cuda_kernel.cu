// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Include associated header file.
#include "../include/cuda_kernel.cuh"





/**
 * Sample CUDA device function which adds an element from array A and array B.
 *
 */
__global__ void MatMul(float *A, float *B, float *C, int M, int N, int K){
   int row = blockIdx.y*blockDim.y + threadIdx.y;
   int col = blockIdx.x*blockDim.x + threadIdx.x;
   if (row < M && col < N) {
    float sum = 0;
    for (int ii = 0; ii < K; ii++) {
      sum += A[row * K + ii] * B[ii * N + col];
    }
    C[row * N + col] = sum;
  }
}



/**
 * Wrapper function for the CUDA kernel function.
 */
void kernel(float *A, float *B, float *C, int M, int N, int K) {
    // Launch CUDA kernel.
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, M*K*sizeof(float));
    cudaMalloc((void**) &d_B, K*N*sizeof(float));
    cudaMalloc((void**) &d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16,16);
    dim3 gridSize(ceil(((float)N)/blockSize.x), 
                  ceil(((float)M)/blockSize.y));

    MatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
}











