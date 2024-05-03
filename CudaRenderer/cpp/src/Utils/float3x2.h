#pragma once

#define MINF __int_as_float(0xff800000)
#define INF  __int_as_float(0x7f800000)

#include <iostream>
#include "cudaUtil.h"
#include "float2x3.h"

#undef max
#undef min

class float2x3;

//////////////////////////////
// float3x2
//////////////////////////////

class float3x2
{
public:

	inline __device__ __host__ float3x2()
	{
	}

	inline __device__ __host__ float3x2(const float values[6])
	{
		m11 = values[0];	m12 = values[1];
		m21 = values[2];	m22 = values[3];
		m31 = values[4];	m32 = values[5];
	}

	inline __device__ __host__ float3x2& operator=(const float3x2& other)
	{
		m11 = other.m11;	m12 = other.m12;
		m21 = other.m21;	m22 = other.m22;
		m31 = other.m31;	m32 = other.m32;
		return *this;
	}

	inline __device__ __host__ float3 operator*(const float2& v) const
	{
		return make_float3(m11*v.x + m12*v.y, m21*v.x + m22*v.y, m31*v.x + m32*v.y);
	}

	inline __device__ __host__ float3x2 operator*(const float t) const
	{
		float3x2 res;
		res.m11 = m11 * t;	res.m12 = m12 * t;
		res.m21 = m21 * t;	res.m22 = m22 * t;
		res.m31 = m31 * t;	res.m32 = m32 * t;
		return res;
	}

	inline __device__ __host__ float& operator()(int i, int j)
	{
		return entries2[i][j];
	}

	inline __device__ __host__ float operator()(int i, int j) const
	{
		return entries2[i][j];
	}

	//inline __device__ __host__ float2x3 getTranspose()
	//{
	//	float2x3 res;
	//	res.m11 = m11; res.m12 = m21; res.m13 = m31;
	//	res.m21 = m12; res.m22 = m22; res.m23 = m32;
	//	return res;
	//}

	inline __device__ __host__ const float* ptr() const {
		return entries;
	}
	inline __device__ __host__ float* ptr() {
		return entries;
	}

	union
	{
		struct
		{
			float m11; float m12;
			float m21; float m22;
			float m31; float m32;
		};

		float entries[6];
		float entries2[3][2];
	};
};