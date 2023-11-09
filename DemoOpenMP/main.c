#include <omp.h>
#include <stdio.h>

int main()
{
	#ifdef _OPENMP
	puts("OpenMP is supported.");
	#endif

	long double start_time, end_time, tick;
	start_time = omp_get_wtime();
	end_time = omp_get_wtime();
	tick = omp_get_wtick();

	printf("end_time - start_time = %.20lf; tick = %.7lf\n", end_time - start_time, tick);

	#pragma omp parallel
	{
		int thread_num = omp_get_thread_num();
		printf("Hello world from thread with thread_num = %d\n", thread_num);
	}

	return 0;
}