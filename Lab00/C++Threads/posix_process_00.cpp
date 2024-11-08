//
// Created by Alessandro on 2019-06-10.
//

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#define NFORKS 5

void do_nothing() {
	printf("This is process: %d\n", getpid());
	exit(11);
}

int main(int argc, char *argv[]) {
	int pid, j, status;

	for (j=0; j < NFORKS; j++) {

		/*** error handling ***/
		if ((pid = fork()) < 0 ) {
			printf ("fork failed with error code= %d\n", pid);
			exit(0);
		}

			/*** this is the child of the fork ***/
		else if (pid ==0) {
			do_nothing();
			exit(1);
		}

			/*** this is the parent of the fork ***/
		else {
			waitpid(pid, &status, 0);
			printf("Status: %d\n", status);
			printf("Status 2: %d\n", WEXITSTATUS(status));
		}
	}
}
