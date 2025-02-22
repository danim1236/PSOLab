#pragma once

#include "../CppPSO/PSOBase.h"

#include <thrust\device_vector.h>
#include <memory>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;
using namespace thrust;

class PSOCuda : public PSOBase
{
public:
	PSOCuda(int numParticles, float *minPositions, float *maxPositions);
	
	void Init();
	void Iterate(int n);

private:
	int _numDevices;
	int _maxThreads;
	int _maxBlocks;

	int _numThreads;
	int _numBlocks;

	device_vector<float> _d_positions;
	device_vector<float> _d_velocities;

	device_vector<float> _d_minPositions;
	device_vector<float> _d_maxPositions;

	device_vector<float> _d_bestPositions;
	device_vector<float> _d_bestFitness;

	device_vector<float> _d_bestGlobalPosition;
	device_vector<float> _d_bestGlobalFitness;

	device_vector<curandState> _d_state;

	void CalculateGeometry();
	void UpdateHost();
};
