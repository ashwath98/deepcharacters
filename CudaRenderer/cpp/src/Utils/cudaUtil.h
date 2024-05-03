#pragma once

#ifndef _CUDA_UTIL_
#define _CUDA_UTIL_

#include <curand.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#include <iostream>

#undef max
#undef min


inline void printCudaError(std::string additionalMessage)
{
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		std::cout  << "cudaCheckError() failed at with " << cudaGetErrorString(err) << std::endl;
		std::cout  << additionalMessage  << std::endl;
		exit(-1);
	}
	return;
}

#define cuRANDErrCheck(ans) { cuRANDAssert((ans), __FILE__, __LINE__); }

inline const char *curandGetErrorString(curandStatus_t rc)
{ 
    switch(rc) { 
    case CURAND_STATUS_SUCCESS: 
	return (std::string("No errors").c_str()); 
    case CURAND_STATUS_VERSION_MISMATCH:  
	return (std::string("Header file and linked library version do not match").c_str());

    case CURAND_STATUS_NOT_INITIALIZED: 
	return (std::string("Generator not initialized").c_str());
    case CURAND_STATUS_ALLOCATION_FAILED: 
	return (std::string("Memory allocation failed").c_str());
    case CURAND_STATUS_TYPE_ERROR: 
	return (std::string("Generator is wrong type").c_str());
    case CURAND_STATUS_OUT_OF_RANGE: 
	return(std::string("Argument out of range").c_str());
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE: 
	return (std::string("Length requested is not a multple of dimension").c_str());
// In CUDA >= 4.1 only 
#if CUDART_VERSION >= 4010 
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: 
	return (std::string("GPU does not have double precision required by MRG32k3a").c_str());
#endif 
    case CURAND_STATUS_LAUNCH_FAILURE: 
	return (std::string("Kernel launch failure").c_str());
    case CURAND_STATUS_PREEXISTING_FAILURE: 
	return (std::string("Preexisting failure on library entry").c_str());
    case CURAND_STATUS_INITIALIZATION_FAILED: 
	return (std::string("Initialization of CUDA failed").c_str());
    case CURAND_STATUS_ARCH_MISMATCH: 
	return (std::string("Architecture mismatch, GPU does not support requested feature").c_str());
    case CURAND_STATUS_INTERNAL_ERROR: 
	return (std::string("Internal library error").c_str());
    default: 
	return (std::string("Unknown error").c_str());
    }
}
 
inline void cuRANDAssert(curandStatus_t code, const char *file, int line, bool abort = true)
{
    if(code != CURAND_STATUS_SUCCESS) 
    {
	std::cerr << "cuRANDAssert: " << curandGetErrorString(code) << ", " << file << ", " << line << std::endl;
	if(abort)
	    while(1); exit(code);
    }
}

// Enable run time assertion checking in kernel code
#define cudaAssert(condition) if (!(condition)) { printf("ASSERT: %s %s\n", #condition, __FILE__); }
//#define cudaAssert(condition)

#if defined(__CUDA_ARCH__)
#define __CONDITIONAL_UNROLL__ #pragma unroll
#else
#define __CONDITIONAL_UNROLL__ 
#endif

#endif
