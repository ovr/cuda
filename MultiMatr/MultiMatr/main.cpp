#include <cuda_runtime.h>

#include <windows.h>
#include <intrin.h>

#include <iostream>
#include <cmath>
#include <iomanip>

#include "kernel.h"
#include "cpu.h"


inline long long gettimeus()
{
	static LARGE_INTEGER ClockPerSecond = { 0 };
	if (ClockPerSecond.QuadPart == 0)
		QueryPerformanceFrequency(&ClockPerSecond);
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);

	return li.QuadPart * 1000000LL / ClockPerSecond.QuadPart;
}


void getInfoCPU()
{
	// Get extended ids.
	int CPUInfo[4] = { -1 };
	__cpuid(CPUInfo, 0x80000000);
	unsigned int nExIds = CPUInfo[0];

	// Get the information associated with each extended ID.
	char CPUBrandString[0x40] = { 0 };
	for (unsigned int i = 0x80000000; i <= nExIds; ++i)
	{
		__cpuid(CPUInfo, i);

		// Interpret CPU brand string and cache information.
		if (i == 0x80000002)
		{
			memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
		}
		else if (i == 0x80000003)
		{
			memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
		}
		else if (i == 0x80000004)
		{
			memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
		}
	}

	std::cout << "CPU: " << CPUBrandString << "\n";
}


void getInfoGPU()
{
	cudaDeviceProp properties;

	cudaGetDeviceProperties(&properties, NULL);
	std::cout << "GPU: " << properties.name << "\n\n";

}



int main()
{
	const size_t N = 2048;
	const size_t size = N * N;

	getInfoCPU();
	getInfoGPU();

	std::cout << "Matrix Multiplication. NxN" << "\n";
	std::cout << "Size: " << N << "x" << N << "\n";
	std::cout << "Type: float" << "\n\n";

	std::cout << "Allocate memory on host... ";
	float *hA = (float*)malloc(size * sizeof(float));
	float *hB = (float*)malloc(size * sizeof(float));
	float *hC = (float*)malloc(size * sizeof(float));
	std::cout << "OK" << "\n";

	std::cout << "Arrays filling on host... ";
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			hA[i * N + j] = std::sinf(i + 1.f);
			hB[i * N + j] = std::sinf(i + 1.f);
		}
	}
	std::cout << "OK" << "\n";

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float *dA, *dB, *dC;

	std::cout << "Allocate memory on device... ";
	cudaMalloc((void**)&dA, size * sizeof(float));
	cudaMalloc((void**)&dB, size * sizeof(float));
	cudaMalloc((void**)&dC, size * sizeof(float));
	std::cout << "OK" << "\n";

	std::cout << "Arrays filling on device... ";
	cudaEventRecord(start, NULL);
	fillMatrixGPU(dA, size, N);
	fillMatrixGPU(dB, size, N);
	std::cout << "OK" << "\n";

	// GPU COMPUTING
	std::cout << "Computing on GPU... ";
	multMatrixGPU(dA, dB, dC, N);
	std::cout << "OK" << "\n";

	std::cout << "Copy data to host memory... ";
	cudaMemcpy(hC, dC, size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, NULL);
	std::cout << "OK" << "\n\n";

	cudaEventSynchronize(stop);
	float gpuTime = 0;
	cudaEventElapsedTime(&gpuTime, start, stop);
	std::cout << "Time(GPU): " << gpuTime << "ms\n\n";


	// CPU COMPUTING
	std::cout << "Computing on CPU... ";
	long long cpuTime = -gettimeus();
	multMatrixCPU(hA, hB, hC, N);
	cpuTime += gettimeus();
	std::cout << "OK" << "\n";

	std::cout << "Time(CPU): " << cpuTime / 1000 << "ms\n";


	std::cout << "Press any key...";


	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	cudaFree(&dA);
	cudaFree(&dB);
	cudaFree(&dC);

	free(hA);
	free(hB);
	free(hC);

	getchar();

	return 0;
}