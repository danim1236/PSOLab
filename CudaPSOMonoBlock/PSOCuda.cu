#include "PSOCuda.cuh"

#include <stdexcept>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/functional.h>

#define NUM_DIMENSIONS 2

__device__ __device_builtin__ void __syncthreads();
__device__ __device_builtin__ float fminf(float x, float y);
__device__ __device_builtin__ float fmaxf(float x, float y);

#define Rand(s, min, max) (curand_uniform(&s)*(max - min) + min)

#define TransferToGlobal()															


__forceinline__ __device__ float EvalBanana(float *position)
{
	float x = position[0];
	float y = position[1];
	float a = y - x * x;
	float b = 1 - x;
	return 100 * a*a + b*b;	
}

__global__ void k_InitPSO(
	int numParticles, 
	float *_positions, 
	float *_velocities, 
	float *_minPositions, 
	float *_maxPositions, 
	float *_bestPositions,
	float *_bestFitness,
	float *_bestGlobalPosition,
	float *_bestGlobalFitness,
	curandState *_s)
{
	__shared__ float positions[1024 * NUM_DIMENSIONS];
	__shared__ float velocities[1024 * NUM_DIMENSIONS];
	__shared__ float bestFitness[1024];
	__shared__ int ptrs[1024];

	int idx = threadIdx.x;
	if (idx >= numParticles)
		bestFitness[idx] = FLT_MAX;
	__syncthreads();

	ptrs[idx] = idx;
	if (idx < numParticles)
	{
		curand_init(threadIdx.x, 0, 0, &_s[idx]);

		int ptr_s = idx * NUM_DIMENSIONS; // posição na memoria shared

		// Calculate randon pos & vel
		float minX = _minPositions[0];
		float minY = _minPositions[0];
		float maxX = _maxPositions[1];
		float maxY = _maxPositions[1];
		positions[ptr_s] = curand_uniform(&_s[idx])*(maxX - minX) + minX;
		positions[ptr_s + 1] = curand_uniform(&_s[idx])*(maxY - minY) + minY;
		velocities[ptr_s] = curand_uniform(&_s[idx])*(maxX - minX) + minX;
		velocities[ptr_s + 1] = curand_uniform(&_s[idx])*(maxY - minY) + minY;

		// Initizalizes local bests
		bestFitness[idx] = EvalBanana(positions + ptr_s);
		__syncthreads();

		// Descobre a melhor
		for (int s = 1024 / 2; s > 0; s /= 2)
		{
			if (idx < s)
			{
				if (bestFitness[ptrs[idx]] > bestFitness[ptrs[idx + s]])
				{
					int tmp = ptrs[idx + s];
					ptrs[idx + s] = ptrs[idx];
					ptrs[idx] = tmp;
				}
			}
			__syncthreads();
		}

		// Transfer to global memory
		for (int d = 0; d < NUM_DIMENSIONS; ++d)										
			_positions[idx + numParticles * d] = positions[idx + numParticles * d];		
		for (int d = 0; d < NUM_DIMENSIONS; ++d)										
			_bestPositions[idx + numParticles * d] = positions[idx + numParticles * d];	
		for (int d = 0; d < NUM_DIMENSIONS; ++d)										
			_velocities[idx + numParticles * d] = velocities[idx + numParticles * d];	
		_bestFitness[idx] = bestFitness[idx];											
		if (idx < NUM_DIMENSIONS)
			_bestGlobalPosition[idx] = positions[ptrs[0] * NUM_DIMENSIONS + idx];
		if (idx == 0)
			_bestGlobalFitness[0] = bestFitness[ptrs[0]];
	}
}

extern "C" __global__ void k_IterateSingleBlock(
	int numIterations,
	int numParticles, 
	float *_positions, 
	float *_velocities, 
	float *_minPositions, 
	float *_maxPositions, 
	float *_bestPositions,
	float *_bestFitness,
	float *_bestGlobalPosition,
	float *_bestGlobalFitness,
	curandState *_s)
{
	__shared__ int ptrs[1024];
	__shared__ float positions[1024 * NUM_DIMENSIONS];
	__shared__ float velocities[1024 * NUM_DIMENSIONS];
	__shared__ float bestFitness[1024];
	__shared__ float bestPositions[1024 * NUM_DIMENSIONS];
	__shared__ float bestGlobalPosition[NUM_DIMENSIONS];
	__shared__ float minPositions[NUM_DIMENSIONS];
	__shared__ float maxPositions[NUM_DIMENSIONS];
	float bestGlobalFitness;

	int p = threadIdx.x;
	if (p < numParticles)
	{
		for (int i = 0; i < NUM_DIMENSIONS; ++i)
			positions[p + numParticles * i] = _positions[p + numParticles * i];
		for (int i = 0; i < NUM_DIMENSIONS; ++i)
			velocities[p + numParticles * i] = _velocities[p + numParticles * i];
		for (int i = 0; i < NUM_DIMENSIONS; ++i)
			bestPositions[p + numParticles * i] = _bestPositions[p + numParticles * i];
		bestFitness[p] = _bestFitness[p];
	}
	if (p < NUM_DIMENSIONS)
	{
		minPositions[p] = _minPositions[p];
		maxPositions[p] = _maxPositions[p];
		bestGlobalPosition[p]  = _bestGlobalPosition[p];
	}
	if (p == 0)
		bestGlobalFitness = _bestGlobalFitness[0];
	__syncthreads();

	while (numIterations-- > 0)
	{
		if (p < numParticles)
		{
			for (int j = 0; j < NUM_DIMENSIONS; ++j)
			{
				float r1 = curand_uniform(&_s[p]);
				float r2 = curand_uniform(&_s[p]);

				float newVelocity = (W * velocities[p * NUM_DIMENSIONS + j]) +
					(C1 * r1 * (bestPositions[p * NUM_DIMENSIONS + j] - positions[p * NUM_DIMENSIONS + j])) +
					(C2 * r2 * (bestGlobalPosition[j] - positions[p * NUM_DIMENSIONS + j]));

				newVelocity = fmaxf(-maxPositions[j], fminf(maxPositions[j], newVelocity));
				velocities[p * NUM_DIMENSIONS + j] = newVelocity;

				float newPosition = positions[p * NUM_DIMENSIONS + j] + newVelocity;
				newPosition = fmaxf(minPositions[j], fminf(maxPositions[j], newPosition));
				positions[p * NUM_DIMENSIONS + j] = newPosition;
			}
			float newFitness = EvalBanana(&positions[p * NUM_DIMENSIONS]);
			if (newFitness < bestFitness[p])
			{
				bestFitness[p] = newFitness;
				for (int j = 0; j < NUM_DIMENSIONS; ++j)
				{
					bestPositions[p * NUM_DIMENSIONS + j] = positions[p * NUM_DIMENSIONS + j];
				}
			}
		}
		__syncthreads();

		// Descobre a melhor
		ptrs[p] = p;
		__syncthreads();
		for (int s = blockDim.x / 2; s > 0; s /= 2)
		{
			if (p < s && p + s < numParticles)
			{
				if (bestFitness[ptrs[p]] > bestFitness[ptrs[p + s]])
				{
					int tmp = ptrs[p + s];
					ptrs[p + s] = ptrs[p];
					ptrs[p] = tmp;
				}
			}
			__syncthreads();
		}

		if (p == 0)
		{
			if (bestFitness[ptrs[0]] < bestGlobalFitness)
			{
				bestGlobalFitness = bestFitness[ptrs[0]];
				for (int j = 0; j < NUM_DIMENSIONS; ++j)
				{
					bestGlobalPosition[j] = positions[ptrs[0] * NUM_DIMENSIONS + j];
				}
			}
		}
		__syncthreads();
	}
	if (p < numParticles)
	{
		for (int i = 0; i < NUM_DIMENSIONS; ++i)
			_positions[p + numParticles * i] = positions[p + numParticles * i];
		for (int i = 0; i < NUM_DIMENSIONS; ++i)
			_velocities[p + numParticles * i] = velocities[p + numParticles * i];
		for (int i = 0; i < NUM_DIMENSIONS; ++i)
			_bestPositions[p + numParticles * i] = bestPositions[p + numParticles * i];
		_bestFitness[p] = bestFitness[p];
	}
	if (p < NUM_DIMENSIONS)
		_bestGlobalPosition[p]  = bestGlobalPosition[p];
	if (p == 0)
		_bestGlobalFitness[0] = bestGlobalFitness;
}


PSOCuda::PSOCuda(int numParticles, float *minPositions, float *maxPositions)
:
PSOBase(numParticles, NUM_DIMENSIONS, minPositions, maxPositions),
_d_positions(_positions.size()),
_d_velocities(_velocities.size()),
_d_minPositions(_minPositions),
_d_maxPositions(_maxPositions),
_d_bestPositions(_bestPositions.size()),
_d_bestFitness(_bestFitness.size()),
_d_bestGlobalPosition(NUM_DIMENSIONS),
_d_bestGlobalFitness(1),
_d_state(numParticles)
{
	CalculateGeometry();
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
}

void PSOCuda::Init()
{
	int threadNumber = pow(2, ceil(log(_numThreads)/log(2)));
	k_InitPSO<<<_numBlocks, threadNumber>>>(_numParticles,
		raw_pointer_cast(_d_positions.data()), 
		raw_pointer_cast(_d_velocities.data()), 
		raw_pointer_cast(_d_minPositions.data()), 
		raw_pointer_cast(_d_maxPositions.data()), 
		raw_pointer_cast(_d_bestPositions.data()),
		raw_pointer_cast(_d_bestFitness.data()),
		raw_pointer_cast(_d_bestGlobalPosition.data()),
		raw_pointer_cast(_d_bestGlobalFitness.data()),
		raw_pointer_cast(_d_state.data()));
	UpdateHost();
}

void PSOCuda::Iterate(int n)
{
	int threadNumber = pow(2, ceil(log(_numThreads)/log(2)));
	k_IterateSingleBlock<<<_numBlocks, threadNumber>>>(n, _numParticles,
		raw_pointer_cast(_d_positions.data()), 
		raw_pointer_cast(_d_velocities.data()), 
		raw_pointer_cast(_d_minPositions.data()), 
		raw_pointer_cast(_d_maxPositions.data()), 
		raw_pointer_cast(_d_bestPositions.data()),
		raw_pointer_cast(_d_bestFitness.data()),
		raw_pointer_cast(_d_bestGlobalPosition.data()),
		raw_pointer_cast(_d_bestGlobalFitness.data()),
		raw_pointer_cast(_d_state.data()));
	UpdateHost();
}

void PSOCuda::UpdateHost()
{
	_positions = _d_positions;
	_velocities = _d_velocities;
	_minPositions = _d_minPositions;
	_maxPositions = _d_maxPositions;
	_bestPositions = _d_bestPositions;
	_bestFitness = _d_bestFitness;
	_bestGlobalPosition = _d_bestGlobalPosition;
}

void PSOCuda::CalculateGeometry()
{
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices < 1)
		throw std::exception("Nenhum dispositivo cuda");

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	int maxThreads = devProp.maxThreadsPerBlock;
	int maxBlocks = 1;

	if (maxThreads * maxBlocks < _numParticles)
		throw std::exception("_maxThreads * _maxBlocks < _numParticles");

	_numThreads = _numParticles / maxBlocks;
	_numThreads = ((_numThreads + 191)/192)*192;
	_numBlocks = (_numParticles + _numThreads - 1) / _numThreads;

	_numThreads = ((_numParticles + 31)/32)*32;
}