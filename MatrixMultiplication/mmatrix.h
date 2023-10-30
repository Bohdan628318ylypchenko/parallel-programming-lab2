#pragma once

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
				double * restrict * restrict * restrict c);

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
				double * restrict * restrict * restrict c);
