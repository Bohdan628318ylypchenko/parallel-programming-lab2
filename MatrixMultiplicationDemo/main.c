#include "mmatrix.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define N_S 4
#define M_S 5
#define K_S 6

#define N_P 1024
#define M_P 2048
#define K_P 3072

#define USAGE "Usage: [v]alidation | [s]ingle-threaded | [m]ulti-threaded thread_count: int"

#define RUN_COUNT 5
#define OUT_SINGLE_THREAD "out-s.txt"
#define OUT_MULTI_THREAD "out-m.txt"

static void mmatrix_validation1(void(*mmatrix)(int, int, int, double **, double **, double ***));
static void mmatrix_validation2(void(*mmatrix)(int, int, int, double **, double **, double ***));
static void mmatrix_validation(void(*mmatrix)(int, int, int, double **, double **, double ***),
							   int n, int m, int k,
							   double ** a, double ** b, double ** c,
							   double ** expected);
static void mmatrix_demo_performance(void(*mmatrix)(int, int, int, double **, double **, double ***), char * outname);

static double ** matrix_alloc(int n, int m);
static void matrix_free(double ** matrix, int m);

static double min_time = DBL_MAX;

int main(int argc, char ** argv)
{
	if (argc < 2)
	{
		puts(USAGE);
		return EXIT_SUCCESS;
	}

	switch (argv[1][0])
	{
		case 'v':

			puts("=== Single thread example 1 ===");
			mmatrix_validation1(mmatrix_1t);
			putchar('\n');
			puts("=== Single thread example 2 ===");
			mmatrix_validation2(mmatrix_1t);
			putchar('\n');
			puts("=== Multi thread example 1 ===");
			mmatrix_validation1(mmatrix_mt);
			putchar('\n');
			puts("=== Multi thread example 2 ===");
			mmatrix_validation2(mmatrix_mt);

			break;
		case 's':

			for (int i = 0; i < RUN_COUNT; i++)
			{
				printf("Single thread run %d | ", i);
				mmatrix_demo_performance(mmatrix_1t, OUT_SINGLE_THREAD);
			}
			printf("Best time: %lf\n", min_time);

			break;

		case 'm':

			if (argc != 3)
			{
				puts(USAGE);
				return EXIT_SUCCESS;
			}

			int n = atoi(argv[2]);
			if (n <= 0)
			{
				printf("Invalid thread count: %s", argv[2]);
				return EXIT_SUCCESS;
			}

			omp_set_num_threads(n);
			for (int i = 0; i < RUN_COUNT; i++)
			{
				printf("Multi thread run %d; n = %d | ", i, n);
				mmatrix_demo_performance(mmatrix_mt, OUT_MULTI_THREAD);
			}
			printf("Best time: %lf\n", min_time);

			break;

		default:
			puts(USAGE);
			break;
	}

	return EXIT_SUCCESS;
}

static void mmatrix_validation1(void(*mmatrix)(int, int, int, double **, double **, double ***))
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

	// Initialize expected
	double ** expected = matrix_alloc(N_S, K_S);
	expected[0][0] = 30; expected[0][1] = 70; expected[0][2] = 110; expected[0][3] = 150; expected[0][4] = 190; expected[0][5] = 230;
	expected[1][0] = 40; expected[1][1] = 95; expected[1][2] = 150; expected[1][3] = 205; expected[1][4] = 260; expected[1][5] = 315;
	expected[2][0] = 50; expected[2][1] = 120; expected[2][2] = 190; expected[2][3] = 260; expected[2][4] = 330; expected[2][5] = 400;
	expected[3][0] = 60; expected[3][1] = 145; expected[3][2] = 230; expected[3][3] = 315; expected[3][4] = 400; expected[3][5] = 485;

	// Validate
	mmatrix_validation(mmatrix, N_S, M_S, K_S, a, b, c, expected);

	// Free resources
	matrix_free(a, N_S);
	matrix_free(b, M_S);
	matrix_free(c, N_S);
	matrix_free(expected, N_S);
}

static void mmatrix_validation2(void(*mmatrix)(int, int, int, double **, double **, double ***))
{
	// Initialize a
	double ** a = matrix_alloc(N_S, N_S);
	a[0][0] = 1; a[0][1] = 1; a[0][2] = 1; a[0][3] = -1;
	a[1][0] = -5; a[1][1] = -3; a[1][2] = -4; a[1][3] = 4;
	a[2][0] = 5; a[2][1] = 1; a[2][2] = 4; a[2][3] = -3;
	a[3][0] = -16; a[3][1] = -11; a[3][2] = -15; a[3][3] = 14;

	// Initialize b
	double ** b = matrix_alloc(N_S, N_S);
	b[0][0] = 7; b[0][1] = -2; b[0][2] = 3; b[0][3] = 4;
	b[1][0] = 11; b[1][1] = 0; b[1][2] = 3; b[1][3] = 4;
	b[2][0] = 5; b[2][1] = 4; b[2][2] = 3; b[2][3] = 0;
	b[3][0] = 22; b[3][1] = 2; b[3][2] = 9; b[3][3] = 8;

	// Initialize c
	double ** c = matrix_alloc(N_S, N_S);

	// Initialize expected
	double ** expected = matrix_alloc(N_S, N_S);
	expected[0][0] = 1; expected[0][1] = 0; expected[0][2] = 0; expected[0][3] = 0;
	expected[1][0] = 0; expected[1][1] = 2; expected[1][2] = 0; expected[1][3] = 0;
	expected[2][0] = 0; expected[2][1] = 0; expected[2][2] = 3; expected[2][3] = 0;
	expected[3][0] = 0; expected[3][1] = 0; expected[3][2] = 0; expected[3][3] = 4;

	// Validate
	mmatrix_validation(mmatrix, N_S, N_S, N_S, a, b, c, expected);

	// Free resources
	matrix_free(a, N_S);
	matrix_free(b, N_S);
	matrix_free(c, N_S);
	matrix_free(expected, N_S);
}

static void mmatrix_validation(void(*mmatrix)(int, int, int, double **, double **, double ***),
							   int n, int m, int k,
							   double ** a, double ** b, double ** c,
							   double ** expected)
{
	// Multiply
	double s_time = omp_get_wtime();
	mmatrix(n, m, k, a, b, &c);
	double e_time = omp_get_wtime();

	// Assertion
	printf("Time: %lf\n", e_time - s_time);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < k; j++)
		{
			printf("| %lf ", c[i][j]);
			if (expected[i][j] != c[i][j])
			{
				printf("\nAssert failed for [%d][%d]\n", i, j);
				break;
			}
		}
		puts("|\n");
	}
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
			b[i][j] = i * j + i + j;
		}
	}

	// Initialize c
	double ** c = matrix_alloc(N_P, K_P);

	// Multiply
	double s_time = omp_get_wtime();
	mmatrix(N_P, M_P, K_P, a, b, &c);
	double e_time = omp_get_wtime();
	double time = e_time - s_time;

	// Assertion
	printf("Time: %lf\n", time);

	// Save time
	if (min_time > time)
		min_time = time;

	// Save result
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