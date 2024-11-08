#include <stdio.h>
#include <omp.h>
int main()
{
	int nthreads, tid;
	printf("Hello parallel world from threads:\n");

	/* set the number of threads (maybe greater than number of 
	   core/processors) */
	omp_set_num_threads(5);

	// fork
	#pragma omp parallel private(tid) default(none)
	{
		tid = omp_get_thread_num();
	  printf("%d\n", tid);
	}
	// implicit join given by }

	printf("Back to the serial world.\n");
}
