#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <iostream>
#include <ctime>

#include "kernel.h"


__global__ void multMatrix(float *a, float *b, float *c, size_t N)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float tmpSum = 0;

	if (row < N && col < N)
	{
		for (int i = 0; i < N; i++)
		{
			tmpSum += a[row * N + i] * b[i * N + col];
		}
	}
	c[row * N + col] = tmpSum;
}


__global__ void  fillMatrix(float *devArr, size_t N)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	for (row = 0; row < N; row++)
	{
		for (col = 0; col < N; col++)
		{
			devArr[row * N + col] = std::sinf(row + 1.f);
		}
	}
}

void multMatrixGPU(float *a, float *b, float *c, size_t N)
{
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);

	if (N * N > 512)
	{
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		blocksPerGrid.x = (unsigned int)(ceil((float)(N) / (float)threadsPerBlock.x));
		blocksPerGrid.y = (unsigned int)(ceil((float)(N) / (float)threadsPerBlock.y));
	}

	//std::cout << "BlockPerGrid: " << blocksPerGrid.x << "x" << blocksPerGrid.y << "\n";
	//std::cout << "ThreadsPerBlock: " << threadsPerBlock.y << "x" << threadsPerBlock.y << "\n\n";

	multMatrix<<< blocksPerGrid, threadsPerBlock >>>(a, b, c, N);
}


void fillMatrixGPU(float *devArr, size_t size, size_t N)
{
	fillMatrix << <1, size >> >(devArr, N);
}