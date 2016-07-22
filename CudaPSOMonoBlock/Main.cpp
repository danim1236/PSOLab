#include "PSOCuda.cuh"

#include <iostream>

int main(int argc, char**argv)
{
	float min[2] = { -10.0, -10.0 };
	float max[2] = { 10.0, 10.0 };

	PSOCuda pso(1024, min, max);
	pso.Init();

	for(int i=0; i < 10; i++) {
		cudaEvent_t start, stop;
		float elapsedTime;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		pso.Iterate(200);

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start,stop);

		float *best = pso.GetBestPosition();
		float *stdDev = pso.GetStdDev();
		std::cout << "x: " << best[0] << "\ty: " << best[1] << 
			"\tstdDev x: " << stdDev[0] << "\tstdDev y: " << stdDev[1] << 
			"\tTime: " << elapsedTime << " ms" << std::endl;
	}
	while (true)
	{

	}
}
