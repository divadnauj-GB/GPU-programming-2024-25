#include <omp.h>
#include <stdio.h>

#define N     1000

int main() {

	int i;
	float a[N], b[N], c[N], d[N];

/* Some initializations */
	for (i = 0; i < N; i++) {
		a[i] = i * 1.5;
		b[i] = i + 22.35;
	}

#pragma omp parallel shared(a, b, c, d) private(i) default(none)
	{

//	  you can use the nowait clause to avoid 
//	  the implied barrier at the end
#pragma omp sections nowait
		{

			// each section is a single thread
#pragma omp section
			for (i = 0; i < N; i++)
				c[i] = a[i] + b[i];

#pragma omp section
			for (i = 0; i < N; i++)
				d[i] = a[i] * b[i];

		}  /* end of sections */

	}  /* end of parallel section */

	for (i = 0; i < N; i++)
		printf("%f ", c[i]);
	printf("\n");

	for (i = 0; i < N; i++)
		printf("%f ", d[i]);
	printf("\n");
}
