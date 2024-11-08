//
// Created by Alessandro on 2019-06-11.
//
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N_ROWS 500
#define N 300
#define N_COLS 700

int main() {
	int i, j, k;
	int firstMatrix[N_ROWS][N];
	int secondMatrix[N][N_COLS];
	int finalMatrix[N_ROWS][N_COLS];
	omp_set_num_threads(8);
	clock_t t = clock();
	// INIT
//#pragma omp parallel for private(j)
//#pragma omp sections nowait
//{
//#pragma omp section
//#pragma omp parallel for
		for (i = 0; i < N_ROWS; ++i) {
			for (j = 0; j < N; ++j) {
				//printf("is it parallel %d, %d ?\n", i, j);
				firstMatrix[i][j] = rand() % 256;
			}
		}

//#pragma omp parallel for private(j)
//#pragma omp section
//#pragma omp parallel for
		for (i = 0; i < N; ++i) {
			for (j = 0; j < N_COLS; ++j) {
				secondMatrix[i][j] = rand() % 256;
			}
		}

	// MUL
	// Initializing elements of matrix mult to 0.
//#pragma omp section
//#pragma omp parallel for
	for (i = 0; i < N_ROWS; ++i) {
		for (j = 0; j < N_COLS; ++j) {
			finalMatrix[i][j] = 0;
		}
	}
//}
	// Multiplying matrix firstMatrix and secondMatrix and storing in array mult.
//#pragma omp parallel for private(i)
	for (i = 0; i < N_ROWS; ++i) {
//#pragma omp parallel for private(j)
		for (j = 0; j < N_COLS; ++j) {
//#pragma omp parallel for shared(finalMatrix) private(k)
			for (k = 0; k < N; ++k) {
				finalMatrix[i][j] += firstMatrix[i][k] *
						secondMatrix[k][j];
			}
		}
	}
	t = clock() - t;
	double time_taken = ((double) t) / CLOCKS_PER_SEC; // in seconds

	printf("The loop took %lf (vs 0.852679) seconds (%d) to execute \n", time_taken, t);
	//print
	/*printf("Output Matrix:\n");
	for (i = 0; i < N_ROWS; ++i) {
		for (j = 0; j < N_COLS; ++j) {
			printf("%d ", finalMatrix[i][j]);
			if (j == N_COLS - 1)
				printf("\n");
		}
	}*/
}
