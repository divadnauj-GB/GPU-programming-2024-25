#include <omp.h>
#include <stdio.h>
#include <time.h>
#define LOOPS 10000

int main() {

	int x;
	x = 0;

	clock_t t = clock();
//#pragma omp parallel num_threads(4) shared(x)
//#pragma omp parallel shared(x) default(none)
	{
#pragma omp parallel for shared(x) default(none)
		for (int i = 0; i < LOOPS; i++) {
			/* Execution one thread at the time */
#pragma omp atomic
				x = x + 1;

		}
	}  /* end of parallel section */
	t = clock() - t;
	double time_taken = ((double) t) / CLOCKS_PER_SEC;
	// in seconds (ref: 0.000028) 0.005667 0.002706

	printf("The loop took %lf seconds (%lu) to execute \n", time_taken, t);
	printf("x = %d\n", x);
}
