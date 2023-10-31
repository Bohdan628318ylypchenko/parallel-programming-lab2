#include "pch.h"
#include "mmatrix.h"

/// <summary>
/// Single thread matrix multiplication.
/// </summary>
/// <param name="n"> row count of a </param>
/// <param name="m"> column count of a = row count of b </param>
/// <param name="k"> column count of b </param>
/// <param name="a"> matrix a as 2d pointer </param>
/// <param name="b"> matrix b as 2d pointer </param>
/// <param name="c"> matrix c as 2d pointer to write result in </param>
void mmatrix_1t(int n, int m, int k,
				const double * const restrict * const restrict a, const double * const restrict * const restrict b,
				double * restrict * restrict * restrict c)
{
	for (int p1 = 0; p1 < n; p1++)
	{
		for (int p2 = 0; p2 < k; p2++)
		{
			(*c)[p1][p2] = 0;
			for (int i = 0; i < m; i++)
			{
				(*c)[p1][p2] += a[p1][i] * b[i][p2];
			}
		}
	}
}

/// <summary>
/// Multi thread matrix multiplication.
/// </summary>
/// <param name="n"> row count of a </param>
/// <param name="m"> column count of a = row count of b </param>
/// <param name="k"> column count of b </param>
/// <param name="a"> matrix a as 2d pointer </param>
/// <param name="b"> matrix b as 2d pointer </param>
/// <param name="c"> matrix c as 2d pointer to write result in </param>
void mmatrix_mt(int n, int m, int k,
				const double * const restrict * const restrict a, const double * const restrict * const restrict b,
				double * restrict * restrict * restrict c)
{
	/*
	 * p1               - private as 1st cycle variable in "for" construct
	 * p2, k            - private as vars declared inside parallel construct
	 * n, m, k, a, b, c - shared as vars declared outside of parallel construct
     */
	int p1;
	#pragma omp parallel for
	for (p1 = 0; p1 < n; p1++)
	{
		for (int p2 = 0; p2 < k; p2++)
		{
			(*c)[p1][p2] = 0;
			for (int i = 0; i < m; i++)
			{
				(*c)[p1][p2] += a[p1][i] * b[i][p2];
			}
		}
	}
}
