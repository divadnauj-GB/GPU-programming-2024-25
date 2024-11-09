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
__global__ void kernel_2dconv(float *A, float *B, float *C, unsigned int ds){
    __shared__ float tile[BLOCK_SIZE+2*RADIUS][BLOCK_SIZE+2*RADIUS];
	__shared__ float kernel[2*RADIUS+1][2*RADIUS+1];

	/* Shared memory tile of size (BLOCK_SIZE+2RADIUS)X(BLOCK_SIZE+2RADIUS)
	_________________________
	|	|				|	|           
	|___|_______________|___|          
	|	|				|	|          
	|	|				|	|          
	|	|				|	|          
	|	|				|	|          
	|___|_______________|___|          
	|	|				|	|          
	|___|_______________|___|          
	*/

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

	int lindex_x = threadIdx.x + RADIUS;
	int lindex_y = threadIdx.y + RADIUS;

	int idx_blk_sz=BLOCK_SIZE;
	int idy_blk_sz=BLOCK_SIZE;

	if(threadIdx.x<=2*RADIUS && threadIdx.y<=2*RADIUS){  //Copy the weighted kernel
		kernel[threadIdx.y][threadIdx.x]=B[threadIdx.y*(2*RADIUS+1)+threadIdx.x];
	}

	if(idx<ds && idy<ds){ //check threads out of bounds

		/*adjust the tile size along x for tiles not divisible by BLOCK_SIZE*/
		if(idx>=((ds-ds%BLOCK_SIZE))){ 
			idx_blk_sz=ds%BLOCK_SIZE;
		}
		/*adjust the tile size along y for tiles not divisible by BLOCK_SIZE*/
		if(idy>=((ds-ds%BLOCK_SIZE))){
			idy_blk_sz=ds%BLOCK_SIZE;
		}

		/*Takes a tile of size BLOCK_SIZE*BLOCK_SIZE*/ 
		/* tile 
		|---|---------------|---|
		|	|				|	|           
		|---|---------------|---|          
		|	|xxxxxxxxxxxxxxx|	|          
		|	|xxxxxxxxxxxxxxx|	|          
		|	|xxxxxxxxxxxxxxx|	|          
		|	|xxxxxxxxxxxxxxx|	|          
		|---|---------------|---|          
		|	|				|	|          
		|---|---------------|---|          
		*/
		tile[lindex_y][lindex_x]=A[idy*ds+idx]; 

		/*Append RADIUS colums on the left and on the right of the tile */
		/* tile 
			|---|---------------|---|
			|	|				|	|           
			|---|---------------|---|          
			|xxx|				|xxx|          
			|xxx|				|xxx|          
			|xxx|				|xxx|          
			|xxx|				|xxx|          
			|---|---------------|---|          
			|	|				|	|          
			|---|---------------|---|          
			*/
		if (threadIdx.x < RADIUS){ // 
			if(idx-RADIUS>=0){
				tile[lindex_y][lindex_x-RADIUS]=A[(idy)*ds+(idx-RADIUS)];
			}else{
				tile[lindex_y][lindex_x-RADIUS]=0; //if out of left bounds extend with zeros
			}
			if((idx+idx_blk_sz)<ds){
				tile[lindex_y][lindex_x+idx_blk_sz]=A[(idy)*ds+(idx+idx_blk_sz)];
			}else{
				tile[lindex_y][lindex_x+idx_blk_sz]=0; //if out of right bounds extend with zeros
			}
		}
		

		/*Append RADIUS Rows on top and the bottom of the tile */
		/* tile 
			|---|---------------|---|
			|	|xxxxxxxxxxxxxxx|	|           
			|---|---------------|---|          
			|	|				|	|          
			|	|				|	|          
			|	|				|	|          
			|	|				|	|          
			|---|---------------|---|          
			|	|xxxxxxxxxxxxxxx|	|          
			|---|---------------|---|          
			*/
		if (threadIdx.y < RADIUS){ // 
			if(idy-RADIUS>=0){
				tile[lindex_y-RADIUS][lindex_x]=A[(idy-RADIUS)*ds+(idx)];
			}else{
				tile[lindex_y-RADIUS][lindex_x]=0;
			}
			if(idy+idy_blk_sz<ds){
				tile[lindex_y+idy_blk_sz][lindex_x]=A[(idy+idy_blk_sz)*ds+(idx)];
			}else{
				tile[lindex_y+idy_blk_sz][lindex_x]=0;
			}
		}
		
		
		/*Fills the top-left and bottom-right corners*/
		/* tile 
			|---|---------------|---|
			|xxx|				|	|           
			|---|---------------|---|          
			|	|				|	|          
			|	|				|	|          
			|	|				|	|          
			|	|				|	|          
			|---|---------------|---|          
			|	|				|xxx|          
			|---|---------------|---|          
			*/
		if (threadIdx.x < RADIUS && threadIdx.y < RADIUS){ // 
			if(idx-RADIUS>=0 && idy-RADIUS>=0){
				tile[lindex_y-RADIUS][lindex_x-RADIUS]=A[(idy-RADIUS)*ds+(idx-RADIUS)];
			}else{
				tile[lindex_y-RADIUS][lindex_x-RADIUS]=0;
			}
			if(idx+idx_blk_sz<ds && idy+idy_blk_sz<ds){
				tile[lindex_y+idy_blk_sz][lindex_x+idx_blk_sz]=A[(idy+idy_blk_sz)*ds+(idx+idx_blk_sz)];
			}else{
				tile[lindex_y+idy_blk_sz][lindex_x+idx_blk_sz]=0;
			}

		}
		
		/*Fills the bottom-left corner*/
		/* tile 
			|---|---------------|---|
			|	|				|	|           
			|---|---------------|---|          
			|	|				|	|          
			|	|				|	|          
			|	|				|	|          
			|	|				|	|          
			|---|---------------|---|          
			|xxx|				|	|          
			|---|---------------|---|          
			*/
		if (threadIdx.x < (RADIUS) && threadIdx.y >= (idy_blk_sz-RADIUS)){ // 
			if(idx-RADIUS>=0 && idy+RADIUS<ds){
				tile[lindex_y+RADIUS][lindex_x-RADIUS]=A[(idy+RADIUS)*ds+(idx-RADIUS)];
			}else{
				tile[lindex_y+RADIUS][lindex_x-RADIUS]=0;
			}
		}

		
		/*Fills the top right corner*/
		/* tile 
			|---|---------------|---|
			|	|				|xxx|           
			|---|---------------|---|          
			|	|				|	|          
			|	|				|	|          
			|	|				|	|          
			|	|				|	|          
			|---|---------------|---|          
			|	|				|	|          
			|---|---------------|---|          
			*/
		if (threadIdx.y < (RADIUS) && threadIdx.x >= (idx_blk_sz-RADIUS)){ // 
			if(idy-RADIUS>=0 && idx+RADIUS<ds){
				tile[lindex_y-RADIUS][lindex_x+RADIUS]=A[(idy-RADIUS)*ds+(idx+RADIUS)];
			}else{
				tile[lindex_y-RADIUS][lindex_x+RADIUS]=0;
			}
		}
		
		__syncthreads();

		/*parallel BLOCK_SIZExBLOCK_SIZE threads performs 2D dot product*/
		float result=0; // 
		for(int row = -RADIUS;row<=RADIUS;row++){ // Compute the 2D dot product
			for(int col = -RADIUS;col<=RADIUS;col++){
				result+=tile[lindex_y+row][lindex_x+col]*kernel[row+RADIUS][col+RADIUS];
			}
		}

		C[idy*ds+idx]=result; //BLOCK_SIZEXBLOCK_SIZE results
	}
}



/**
 * Wrapper function for the CUDA kernel function.
 */
void kernel(float *A, float *B, float *C, int ds) {
    // Launch CUDA kernel.
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, ds*ds*sizeof(float));
    cudaMalloc((void**) &d_B, (2*RADIUS+1)*(2*RADIUS+1)*sizeof(float));
    cudaMalloc((void**) &d_C, ds*ds*sizeof(float));

    cudaMemcpy(d_A, A, ds*ds*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (2*RADIUS+1)*(2*RADIUS+1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, ds*ds*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 gridSize((ds)/BLOCK_SIZE+1,(ds)/BLOCK_SIZE+1,1);

    kernel_2dconv<<<gridSize, blockSize>>>(d_A, d_B, d_C, ds);
    
    cudaMemcpy(C, d_C, ds*ds*sizeof(float), cudaMemcpyDeviceToHost);
}











