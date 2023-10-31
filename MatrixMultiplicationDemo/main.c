#include "mmatrix.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N_S 4
#define M_S 5
#define K_S 6

#define N_P 1024
#define M_P 2048
#define K_P 3072

#define USAGE "Usage: s[ingle-threaded] | m[ulti-threaded] thread-count:int\n"
#define FLAG_SINGLE_THREAD 's'
#define FLAG_MULTI_THREAD  'm'
#define OUT_SINGLE_THREAD "out-s.txt"
#define OUT_MULTI_THREAD "out-m.txt"

static void mmatrix_demo_simple(void(*mmatrix)(int, int, int, double **, double **, double ***));
static void mmatrix_demo_performance(void(*mmatrix)(int, int, int, double **, double **, double ***), char * outname);

static double ** matrix_alloc(int n, int m);
static void matrix_free(double ** matrix, int m);

int main(int argc, char ** argv)
{
	if (argc < 2)
	{
		puts(USAGE);
		return EXIT_SUCCESS;
	}

	switch (argv[1][0])
	{
		case FLAG_SINGLE_THREAD:
			if (argc != 2)
			{
				puts(USAGE);
				return EXIT_SUCCESS;
			}
			puts("=== Single-threaded simple demo ===");
			mmatrix_demo_simple(mmatrix_1t);
			puts("=== Single-threaded simple demo end ===\n");
			puts("\n=== Single-threaded performance demo ===");
			mmatrix_demo_performance(mmatrix_1t, OUT_SINGLE_THREAD);
			puts("=== Single-threaded performance demo end ===");
			break;
		case FLAG_MULTI_THREAD:
			if (argc != 3)
			{
				puts(USAGE);
				return EXIT_SUCCESS;
			}
			int num_threads = atoi(argv[2]);
			if (num_threads <= 0)
			{
				printf("Invalid thread count: %s\n", argv[2]);
				return EXIT_SUCCESS;
			}
			omp_set_num_threads(num_threads);
			puts("=== Multi-threaded simple demo ===");
			mmatrix_demo_simple(mmatrix_mt);
			puts("=== Multi-threaded simple demo end ===\n");
			puts("\n=== Multi-threaded performance demo ===");
			mmatrix_demo_performance(mmatrix_mt, OUT_MULTI_THREAD);
			puts("=== Multi-threaded performance demo end ===");
			break;
		default:
			puts(USAGE);
			return EXIT_SUCCESS;
	}

	return EXIT_SUCCESS;
}

static void mmatrix_demo_simple(void(*mmatrix)(int, int, int, double **, double **, double ***))
{
	// Initialize a
	double ** a = matrix_alloc(N_S, M_S);
	for (int i = 0; i < N_S; i++)
	{
		for (int j = 0; j < M_S; j++)
		{
			a[i][j] = i + j;
		}
	}

	// Initialize b
	double ** b = matrix_alloc(M_S, K_S);
	for (int i = 0; i < M_S; i++)
	{
		for (int j = 0; j < K_S; j++)
		{
			b[i][j] = i * j + i + j;
		}
	}

	// Initialize c
	double ** c = matrix_alloc(N_S, K_S);

	// Multiply
	double s_time = omp_get_wtime();
	mmatrix(N_S, M_S, K_S, a, b, &c);
	double e_time = omp_get_wtime();

	// Assertion
	printf("Time: %lf\n", e_time - s_time);
	double result[N_S][K_S] = { { 30, 70, 110, 150, 190, 230  },
						        { 40, 95, 150, 205, 260, 315  },
						        { 50, 120, 190, 260, 330, 400 },
						        { 60, 145, 230, 315, 400, 485 } };
	int assertion_flag = 1;
	for (int i = 0; i < N_S; i++)
	{
		for (int j = 0; j < K_S; j++)
		{
			printf("| %lf ", result[i][j]);
			if (result[i][j] != c[i][j])
			{
				printf("\nAssert failed for [%d][%d]\n", i, j);
				assertion_flag = 0;
				break;
			}
		}
		puts("|\n");
	}
	if (assertion_flag)
		printf("Assert succeeded\n");
	else
		printf("Assert failed\n");

	// Free resources
	matrix_free(a, N_S);
	matrix_free(b, M_S);
	matrix_free(c, N_S);
}

static void mmatrix_demo_performance(void(*mmatrix)(int, int, int, double **, double **, double ***), char * outname)
{
	// Initialize a
	double ** a = matrix_alloc(N_P, M_P);
	for (int i = 0; i < N_P; i++)
	{
		for (int j = 0; j < M_P; j++)
		{
			a[i][j] = i + j;
		}
	}

	// Initialize b
	double ** b = matrix_alloc(M_P, K_P);
	for (int i = 0; i < M_P; i++)
	{
		for (int j = 0; j < K_P; j++)
		{
			b[i][j] = i + j;
		}
	}

	// Initialize c
	double ** c = matrix_alloc(N_P, K_P);

	// Multiply
	double s_time = omp_get_wtime();
	mmatrix(N_P, M_P, K_P, a, b, &c);
	double e_time = omp_get_wtime();

	// Assertion
	printf("Time: %lf\n", e_time - s_time);

	// Save
	FILE * f;
	fopen_s(&f, outname, "w");
	for (int i = 0; i < N_P; i++)
	{
		for (int j = 0; j < K_P; j++)
		{
			fprintf(f, "| %lf ", c[i][j]);
		}
		fputs("|\n", f);
	}
	fputc('\n', f);
	fclose(f);

	// Free resources
	matrix_free(a, N_P);
	matrix_free(b, M_P);
	matrix_free(c, N_P);
}

static double ** matrix_alloc(int n, int m)
{
	double ** matrix = (double **)malloc(n * sizeof(double *));
	for (int i = 0; i < n; i++)
	{
		matrix[i] = (double *)malloc(m * sizeof(double));
	}
	return matrix;
}

static void matrix_free(double ** matrix, int row_count)
{
	for (int i = 0; i < row_count; i++)
	{
		free(matrix[i]);
	}
	free(matrix);
}