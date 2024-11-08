# Laboratory 01

Write your first CUDA program to familiarize yourself with the GPU programming toolchain as well as to get familiar with the CUDA programming model, including concepts such as grids, blocks,
thread indexing

The overall structure of a CUDA program that uses the GPU for computation is as follows:

1. Define the code that will run on the device in a separate function, called the kernel function.
2. In the main program running on the host’s CPU:
a. allocate memory on the host for the data arrays.
b. initialize the data arrays in the host’s memory.
c. allocate separate memory on the GPU device for the data arrays.
d. copy data arrays from the host memory to the GPU device memory.
3. On the GPU device, execute the kernel function that computes new data values given the original
arrays. Specify how many blocks and threads per block to use for this computation.
4. After the kernel function completes, copy the computed values from the GPU device memory back
to the host’s memory.

## **Exercise 1: Vector addition**

1. Write a Kernel for adding 2 Vectors
2. Allocate memory for arrays A, B, and C on the GPU.
3. Generate random values for arrays A and B.
4. Write a CUDA kernel function to perform element-wise addition.
5. Launch the kernel with an appropriate grid and block size.
6. Copy the result back from the GPU to the CPU and verify the correctness of the computation.
7. Make the computation of arrays with sizes 1000, 10000, and 1000000 using different parallel configurations of Blocks and threads per block.
    - Record the execution time when executing on the CPU only
    - Record the time required to transfer data between the CPU and GPU
    - Record the time required to execute a kernel on the GPU
    - Do you identify any bottlenecks? Does it make sense to use a GPU to compute all vector sizes?
8. For vectors of size 10000, make the required adjustments to execute 32, 64, 128, 256, 512, 1024, 2048,
4096, and 8192 threads per block.

## **Exercise 2: Matrix multiplication**

*Write a CUDA program to perform matrix multiplication. Given two matrices A (of size MxK) and B (of size KxN), compute their matrix product and store the result in a third matrix C (of size MxN) Implement a CUDA kernel function to perform matrix multiplication in parallel.*

1. Allocate memory for matrices A, B, and C on the GPU.
2. Generate values for matrices A and B.
3. Write a CUDA kernel function to perform matrix multiplication.
4. Launch the kernel with an appropriate grid and block size.
5. Copy the result back from the GPU to the CPU and verify the correctness of the computation.

6. Make the computation of square matrices with sizes 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
and 32768
    - Record the execution time when executing on the CPU only
    - Record the time required to transfer data between the CPU and GPU
    - Record the time required to execute a kernel on the GPU
    - Do you identify any bottlenecks? Does it make sense to use a GPU to compute all matrix multiplication sizes?

7. For square matrices of size 4096, make the required adjustments to execute 32, 64, 128, 256, 512, 1024,and 2048 threads per block.
    - Does the performance change?

8. For square matrices of size 4096, make the required adjustments to execute the whole operation using
only 256, 512, 1024, or 2048 threads.
