#include "PSOCpp.h"

#include <iostream>
#include <time.h>

int main(int argc, char**argv)
{
	float min[2] = { -10.0, -10.0 };
	float max[2] = { 10.0, 10.0 };

	PSOCpp pso(51200, 2, min, max);
	pso.Init();

	while(1) {
		clock_t start = clock();

		pso.Iterate(50);

		clock_t stop = clock();

		float elapsedTime = 1000.0 * (stop - start) / CLOCKS_PER_SEC;

		float *best = pso.GetBestPosition();
		float *stdDev = pso.GetStdDev();
		std::cout << "x: " << best[0] << "\ty: " << best[1] << 
			"\tstdDev x: " << stdDev[0] << "\tstdDev y: " << stdDev[1] << 
			"\tTime: " << elapsedTime << " ms" << std::endl;
	}
}
