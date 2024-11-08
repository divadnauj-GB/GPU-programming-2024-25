/******************************************************************************/
/*
    Discussion:

    This program should be run on at least two processes.
    Any processes beyond the first two will not be given any work.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    11 October 2007

  Author:

    John Burkardt

  Reference:

    William Gropp, Ewing Lusk, Anthony Skjellum,
    Using MPI: Portable Parallel Programming with the
    Message-Passing Interface,
    Second Edition,
    MIT Press, 1999,
    ISBN: 0262571323.
*/
# include <mpi.h>
# include <stdio.h>
# include <stdlib.h>
# include <time.h>

/******************************************************************************/

int main ( int argc, char *argv[] ) {
	int count;
	float data[200];
	int dest;
	int i;
	int ierr;
	int num_procs;
	int rank;
	int source;
	MPI_Status status;
	int tag;
	float value[200];
/*
  Initialize MPI.
*/
	ierr = MPI_Init ( &argc, &argv );
/*
  Determine this process's rank.
*/
	ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
/*
  Determine the number of available processes.
*/
	ierr = MPI_Comm_size ( MPI_COMM_WORLD, &num_procs );
/*
  Have Process 0 say hello.
*/
	if ( rank == 0 )
	{
		printf ( "\n" );
		printf ( "Vector Pass:\n" );
		printf ( "  C version\n" );
		printf ( "  An MPI example program.\n" );
		printf ( "  The number of processes available is %d\n", num_procs );
	}
/*
  Process 0 expects up to 200 real values, from any source.
*/
	if ( rank == 0 )
	{
		source = 1;
		tag = 55;

		ierr = MPI_Recv ( value, 100, MPI_FLOAT, MPI_ANY_SOURCE, tag,
						  MPI_COMM_WORLD, &status );

		ierr = MPI_Get_count ( &status, MPI_FLOAT, &count );

		printf ( "P:%d Got %d elements.\n", rank, count );

		source = 2;
		ierr = MPI_Recv ( value+100, 100, MPI_FLOAT, MPI_ANY_SOURCE, tag,
						  MPI_COMM_WORLD, &status );

		ierr = MPI_Get_count ( &status, MPI_FLOAT, &count );

		printf ( "P:%d Got %d elements.\n", rank, count );

		for (int i = 0; i < 200; ++i) {
			if (value[i] == 5)
				printf ( "P:%d value[%d] = %f\n", rank, i, value[i] );
		}
	}
/*
  Processes 1 and 2 send 100 real values to process 0.
*/
	else if (( rank == 1 ) || ( rank == 2))
	{
		printf ( "\n" );
		printf ( "P:%d - setting up data to send to process 0.\n", rank );

		for ( i = 0; i < 100; i++ )
		{
			data[i] = i;
		}

		dest = 0;
		tag = 55;
		ierr = MPI_Send ( data, 100, MPI_FLOAT, dest, tag, MPI_COMM_WORLD );
	}
/*
  Any other process is idle.
*/
	else
	{
		printf ( "\n" );
		printf ( "P:%d - MPI has no work for me!\n", rank );
	}
/*
  Terminate MPI.
*/
	ierr = MPI_Finalize ( );
/*
  Terminate.
*/
	if ( rank == 0 )
	{
		printf ( "\n" );
		printf ( "Vector Pass:\n" );
		printf ( "  Normal end of execution.\n" );
		printf ( "\n" );
	}
	return 0;
}
