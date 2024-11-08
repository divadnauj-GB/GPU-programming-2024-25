#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHUNKSIZE 100000
#define N     100000

int main() {

	int i, chunk;
	float a[N], b[N], c[N];

/* Some initializations */
	for (i = 0; i < N; i++)
		a[i] = b[i] = i * 1.0;
	chunk = CHUNKSIZE;

	clock_t t = clock();
#pragma omp parallel shared(a, b, c, chunk) private(i) default(none)
	{

#pragma omp for schedule(dynamic, chunk) nowait
		for (i = 0; i < N; i++)
			c[i] = a[i] + b[i];

	}  /* end of parallel section */
	t = clock() - t;
	double time_taken = ((double) t) / CLOCKS_PER_SEC; // in seconds

	printf("The loop took %lf seconds (%d) to execute \n", time_taken, t);
	/*for (i=0; i < N; i++) {
	  printf("%d: %f\n", i, c[i]);
	}*/
}
