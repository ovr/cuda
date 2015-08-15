#include "cpu.h"
#include <iostream>


void multMatrixCPU(float *a, float *b, float *c, size_t N)
{
	// Transpose array B
	//std::cout << "\n";
	float *transB = new float*[N*N];
	
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			transB[i * N + j] = b[j * N + i];
		}
	}

	/*for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			std::cout << transB[i * N + j] << " ";
		}
		std::cout << "\n";
	}*/

	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			c[i * N + j] = 0;
			for (size_t k = 0; k < N; k++)
			{
				c[i * N + j] += a[i * N + k] * transB[k * N + j];
			}
		}
	}
}
