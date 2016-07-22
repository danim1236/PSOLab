#pragma once


#include "PSOBase.h"


class PSOCpp : public PSOBase
{
public:
	PSOCpp(int numParticles, int numDimensions, float *minPositions, float *maxPositions)
	:	PSOBase(numParticles, numDimensions, minPositions, maxPositions)
	{
	}

	void Init();
	void Iterate(int n);
};

