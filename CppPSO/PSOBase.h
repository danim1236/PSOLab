#pragma once

#include <thrust\host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

using namespace thrust;
using namespace thrust::cuda::experimental;

#define W (float)0.729
#define C1 (float)1.49445
#define C2 (float)1.49445

class PSOBase
{
public:
	PSOBase(int numParticles, int numDimensions,
			float *minPositions, float *maxPositions);

	virtual void Init() = 0;
	virtual void Iterate(int n) = 0;
	
	inline float Eval(float *position)
	{
		float x = position[0];
		float y = position[1];
		float a = y - x*x;
		float b = 1 -x;
		return 100 * a*a + b*b;
	}

	float* GetPosition(int particle) { return &_positions[particle * _numDimensions]; }
	float* GetBestPosition() { return &_bestGlobalPosition[0]; }
	float* GetVelocity(int particle) { return &_velocities[particle * _numDimensions]; }
	float* GetStdDev();

protected:
	int _numParticles;
	int _numDimensions;

	host_vector<float, pinned_allocator<float>> _positions;
	host_vector<float, pinned_allocator<float>> _velocities;

	host_vector<float, pinned_allocator<float>> _minPositions;
	host_vector<float, pinned_allocator<float>> _maxPositions;

	host_vector<float, pinned_allocator<float>> _bestPositions;
	host_vector<float, pinned_allocator<float>> _bestFitness;

	host_vector<float, pinned_allocator<float>> _bestGlobalPosition;
	host_vector<float, pinned_allocator<float>> _bestGlobalFitness;
};

