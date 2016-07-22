#include "PSOCuda.cuh"

#include <stdexcept>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/functional.h>
#include <thrust/extrema.h>

#define NUM_DIMENSIONS 2


extern "C" __device__ __device_builtin__ void __syncthreads();
extern "C" __device__ __device_builtin__ float fminf(float x, float y);
extern "C" __device__ __device_builtin__ float fmaxf(float x, float y);
extern "C" __device__ __device_builtin__ unsigned int __uAtomicInc(unsigned int *address, unsigned int val);
extern "C"  __device__ __device_builtin__ void __threadfence_system(void);

#define Rand(s, min, max) (curand_uniform(&s)*(max - min) + min)

__constant__ float _c_minPosition[NUM_DIMENSIONS];
__constant__ float _c_maxPosition[NUM_DIMENSIONS];

__forceinline__ __device__ float EvalBanana(float *position)
{
	float x = position[0];
	float y = position[1];
	float a = y - x * x;
	float b = 1 - x;
	return 100 * a*a + b*b;	
}

__forceinline__ __device__ void WaitForIncBlocks(unsigned int *itCount, int it, unsigned int max)
{
	if (threadIdx.x == 0)
	{
		__threadfence_system();
		__uAtomicInc(&itCount[blockIdx.x], max + 1);
		__threadfence_system();
		int cont = 1;
		while (cont)
		{
			cont = 0;
			for (int i = 0; i < gridDim.x; ++i)
			{
				if (itCount[blockIdx.x] != it + 1)
				{
					cont = 1;
					break;
				}
			}
		}
	}
	__syncthreads();
}


__global__ void k_InitPSO(
	int numParticles, 
	float *_positions, 
	float *_velocities, 
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
	int gidx = blockDim.x * blockIdx.x + idx;
	if (gidx >= numParticles)
		bestFitness[idx] = FLT_MAX;
	__syncthreads();

	ptrs[idx] = idx;
	if (idx < numParticles)
	{
		curand_init(threadIdx.x, 0, 0, &_s[idx]);

		int ptr_s = idx * NUM_DIMENSIONS; // posição na memoria shared

		// Calculate randon pos & vel
		float minX = _c_minPosition[0];
		float minY = _c_minPosition[0];
		float maxX = _c_maxPosition[1];
		float maxY = _c_maxPosition[1];
		positions[ptr_s] = curand_uniform(&_s[idx])*(maxX - minX) + minX;
		positions[ptr_s + 1] = curand_uniform(&_s[idx])*(maxY - minY) + minY;
		velocities[ptr_s] = curand_uniform(&_s[idx])*(maxX - minX) + minX;
		velocities[ptr_s + 1] = curand_uniform(&_s[idx])*(maxY - minY) + minY;

		// Initizalizes local bests
		bestFitness[idx] = EvalBanana(positions + ptr_s);
	}
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

	if (gidx < numParticles)
	{
		// Transfer to global memory
		for (int d = 0; d < NUM_DIMENSIONS; ++d)										
			_positions[gidx * NUM_DIMENSIONS + d] = positions[idx * NUM_DIMENSIONS + d];
		for (int d = 0; d < NUM_DIMENSIONS; ++d)
			_bestPositions[gidx * NUM_DIMENSIONS + d] = positions[idx * NUM_DIMENSIONS + d];
		for (int d = 0; d < NUM_DIMENSIONS; ++d)
			_velocities[gidx * NUM_DIMENSIONS + d] = velocities[idx * NUM_DIMENSIONS + d];
		_bestFitness[gidx] = bestFitness[idx];
		if (idx < NUM_DIMENSIONS)
			_bestGlobalPosition[blockIdx.x * NUM_DIMENSIONS + idx] = positions[ptrs[0] * NUM_DIMENSIONS + idx];
		if (idx == 0)
			_bestGlobalFitness[blockIdx.x] = bestFitness[ptrs[0]];
	}
}

extern "C" __global__ void k_IterateMultiBlock(
	int numParticles, 
	float *_positions, 
	float *_velocities, 
	float *_bestPositions,
	float *_bestFitness,
	float *_bestGlobalPosition,
	float *_bestGlobalFitness,
	curandState *_s)
{
	__shared__ int ptrs[1024];
	__shared__ float positions[1024 * NUM_DIMENSIONS];
	__shared__ float bestFitness[1024];
	__shared__ float bestPositions[1024 * NUM_DIMENSIONS];
	float bestGlobalFitness;

	int p = threadIdx.x;
	int block = blockIdx.x;
	int gp = blockDim.x * block + p;
	if (gp < numParticles)
	{
		for (int i = 0; i < NUM_DIMENSIONS; ++i)
			positions[p * NUM_DIMENSIONS + i] = _positions[gp * NUM_DIMENSIONS + i];
		for (int i = 0; i < NUM_DIMENSIONS; ++i)
			bestPositions[p * NUM_DIMENSIONS + i] = _bestPositions[gp * NUM_DIMENSIONS + i];
		bestFitness[p] = _bestFitness[gp];
	}

	if (p == 0)
		bestGlobalFitness = _bestGlobalFitness[0];
	else if (gp >= numParticles)
		bestFitness[p] = FLT_MAX;
	__syncthreads();
	if (gp < numParticles)
	{
		for (int j = 0; j < NUM_DIMENSIONS; ++j)
		{
			float r1 = curand_uniform(&_s[p]);
			float r2 = curand_uniform(&_s[p]);

			float newVelocity = (W * _velocities[gp * NUM_DIMENSIONS + j]) +
				(C1 * r1 * (bestPositions[p * NUM_DIMENSIONS + j] - positions[p * NUM_DIMENSIONS + j])) +
				(C2 * r2 * (_bestGlobalPosition[block * NUM_DIMENSIONS + j] - positions[p * NUM_DIMENSIONS + j]));

			newVelocity = fmaxf(_c_minPosition[j], fminf(_c_maxPosition[j], newVelocity));
			_velocities[gp * NUM_DIMENSIONS + j] = newVelocity;

			float newPosition = positions[p * NUM_DIMENSIONS + j] + newVelocity;
			newPosition = fmaxf(_c_minPosition[j], fminf(_c_maxPosition[j], newPosition));
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
		if (p < s)
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
				_bestGlobalPosition[block * NUM_DIMENSIONS + j] = positions[ptrs[0] * NUM_DIMENSIONS + j];
			}
		}
	}
	__syncthreads();
	
	if (gp < numParticles)
	{
		int ptr_g = gp * NUM_DIMENSIONS;
		int ptr_s = p * NUM_DIMENSIONS;

		for (int d = 0; d < NUM_DIMENSIONS; ++d)
			_positions[ptr_g + d] = positions[ptr_s + d];
		for (int d = 0; d < NUM_DIMENSIONS; ++d)
			_bestPositions[ptr_g + d] = bestPositions[ptr_s + d];
		_bestFitness[gp] = bestFitness[p];
	}
	if (p == 0)
		_bestGlobalFitness[block] = bestGlobalFitness;
}

__global__ void k_minimum(int _numBlocks, float *_position, float *_fitness)
{
	__shared__ float fitness[1024];
	__shared__ int ptrs[1024];

	int idx = threadIdx.x;
	ptrs[idx] = idx;
	if (idx >= _numBlocks)
		fitness[idx] = FLT_MAX;
	__syncthreads();

	if (idx < _numBlocks)
		fitness[idx] = _fitness[idx];
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s /= 2)
	{
		if (idx < s)
		{
			if (fitness[ptrs[idx]] > fitness[ptrs[idx + s]])
			{
				int tmp = ptrs[idx + s];
				ptrs[idx + s] = ptrs[idx];
				ptrs[idx] = tmp;
			}
		}
		__syncthreads();
	}
	if (idx < NUM_DIMENSIONS)
		_position[idx] = _position[ptrs[0] * NUM_DIMENSIONS + idx];
	if (idx == 0)
		_fitness[0] = _fitness[ptrs[0]];
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
_d_state(numParticles)
{
	CalculateGeometry();
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	_d_bestGlobalPosition.resize(NUM_DIMENSIONS * _numBlocks);
	_d_bestGlobalFitness.resize(_numBlocks);
	_bestGlobalPosition.resize(NUM_DIMENSIONS * _numBlocks);
	_bestGlobalFitness.resize(_numBlocks);
	cudaMemcpyToSymbol(_c_minPosition, _minPositions.data(), _minPositions.size() * sizeof(float));
	cudaMemcpyToSymbol(_c_maxPosition, _maxPositions.data(), _maxPositions.size() * sizeof(float));

}

void PSOCuda::Init()
{
	int threadNumber = pow(2, ceil(log(_numThreads)/log(2)));
	int blockNumber = pow(2, ceil(log(_numBlocks)/log(2)));
	k_InitPSO<<<_numBlocks, threadNumber>>>(_numParticles,
		raw_pointer_cast(_d_positions.data()), 
		raw_pointer_cast(_d_velocities.data()), 
		raw_pointer_cast(_d_bestPositions.data()),
		raw_pointer_cast(_d_bestFitness.data()),
		raw_pointer_cast(_d_bestGlobalPosition.data()),
		raw_pointer_cast(_d_bestGlobalFitness.data()),
		raw_pointer_cast(_d_state.data()));
	cudaDeviceSynchronize();
	k_minimum<<<1, blockNumber>>>(_numBlocks,
		raw_pointer_cast(_d_bestGlobalPosition.data()),
		raw_pointer_cast(_d_bestGlobalFitness.data()));
	UpdateHost();
}

void PSOCuda::Iterate(int n)
{
	int threadNumber = pow(2, ceil(log(_numThreads)/log(2)));
	int blockNumber = pow(2, ceil(log(_numBlocks)/log(2)));
	for (int i = 0; i < n; ++i)
	{
		k_IterateMultiBlock<<<_numBlocks, threadNumber>>>(_numParticles,
			raw_pointer_cast(_d_positions.data()), 
			raw_pointer_cast(_d_velocities.data()), 
			raw_pointer_cast(_d_bestPositions.data()),
			raw_pointer_cast(_d_bestFitness.data()),
			raw_pointer_cast(_d_bestGlobalPosition.data()),
			raw_pointer_cast(_d_bestGlobalFitness.data()),
			raw_pointer_cast(_d_state.data()));
		cudaDeviceSynchronize();
		k_minimum<<<1, blockNumber>>>(_numBlocks,
			raw_pointer_cast(_d_bestGlobalPosition.data()),
			raw_pointer_cast(_d_bestGlobalFitness.data()));
	}
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
	_bestGlobalFitness = _d_bestGlobalFitness;
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
	int maxBlocks = devProp.multiProcessorCount;

	//if (maxThreads * maxBlocks < _numParticles)
	//	throw std::exception("_maxThreads * _maxBlocks < _numParticles");

	_numThreads = _numParticles / maxBlocks;
	_numThreads = std::min(((_numThreads + 191)/192)*192, maxThreads);
	_numBlocks = (_numParticles + _numThreads - 1) / _numThreads;
}