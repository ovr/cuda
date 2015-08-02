#include "cpu.h"

void multMatrixCPU(float *a, float *b, float *c, size_t N)
{
	float sum;
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			sum = 0.f;
			for (size_t k = 0; k < N; k++)
			{
				sum += a[i * N + k] * b[k * N + j];
			}
			c[i * N + j] = sum;
		}
	}
}