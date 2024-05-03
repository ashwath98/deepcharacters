//==============================================================================================//
// Classname:
//      BlockReduce
//
//==============================================================================================//
// Description:
//      Fast CUDA-based summing over huge arrays
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include "cudaUtil.h"
#undef max
#undef min

//==============================================================================================//

#include "cuda_SimpleMatrixUtil.h"

//==============================================================================================//

#define FLOAT_EPSILON 0.0001f

#ifndef BYTE
#define BYTE unsigned char
#endif

#define MINF __int_as_float(0xff800000)

//==============================================================================================//

extern __shared__ float bucket[];

//==============================================================================================//

__inline__ __device__ float warpReduce(volatile float val)
{
	int offset = 32 >> 1;
	while (offset > 0)
	{
		val = val + __shfl_down_sync(val, offset, 32);
		offset = offset >> 1;
	}
	return val;
}

//==============================================================================================//

inline __device__ void blockReduce(volatile float* sdata, int threadIdx, unsigned int threadsPerBlock)
{
	#pragma unroll
	for(unsigned int stride = threadsPerBlock/2 ; stride >= 32; stride/=2)
	{
		if(threadIdx < stride) sdata[threadIdx] = sdata[threadIdx] + sdata[threadIdx+stride];

		__syncthreads();
	}

	sdata[threadIdx] = warpReduce(sdata[threadIdx]);

	__syncthreads();
}

//==============================================================================================//

inline __device__ void scanPart1(unsigned int threadIdx, unsigned int blockIdx, unsigned int threadsPerBlock, float* d_output)
{
	__syncthreads();
	blockReduce(bucket, threadIdx, threadsPerBlock);
	if(threadIdx == 0) d_output[blockIdx] = bucket[0];
}

//==============================================================================================//

inline __device__ void scanPart2(unsigned int threadIdx, unsigned int threadsPerBlock, unsigned int blocksPerGrid, float* d_tmp)
{
	if(threadIdx < blocksPerGrid) bucket[threadIdx] = d_tmp[threadIdx];
	else						  bucket[threadIdx] = 0.0f;
	
	__syncthreads();
	blockReduce(bucket, threadIdx, threadsPerBlock);
	__syncthreads();
}

//==============================================================================================//
