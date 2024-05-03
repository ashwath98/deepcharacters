
#pragma once

#define MINF __int_as_float(0xff800000)
#define INF  __int_as_float(0x7f800000)

#include <iostream>
#include "cudaUtil.h"
#include "float3x2.h"

#undef max
#undef min

//////////////////////////////
// float2x3
//////////////////////////////

class float2x3
{
public:

	inline __device__ __host__ float2x3()
	{
	}

	inline __device__ __host__ float2x3(const float values[6])
	{
		m11 = values[0];	m12 = values[1];	m13 = values[2];
		m21 = values[3];	m22 = values[4];	m23 = values[5];
	}

	inline __device__ __host__ float2x3(const float2x3& other)
	{
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;
		m21 = other.m21; 	m22 = other.m22;	m23 = other.m23;
	}

	inline __device__ __host__ float2x3& operator=(const float2x3 &other)
	{
		m11 = other.m11;	m12 = other.m12; m13 = other.m13;
		m21 = other.m21;	m22 = other.m22; m23 = other.m23;
		return *this;
	}

	inline __device__ __host__ float2 operator*(const float3 &v) const
	{
		return make_float2(m11*v.x + m12*v.y + m13*v.z, m21*v.x + m22*v.y + m23*v.z);
	}

	//! matrix scalar multiplication
	inline __device__ __host__ float2x3 operator*(const float t) const
	{
		float2x3 res;
		res.m11 = m11 * t;	res.m12 = m12 * t;	res.m13 = m13 * t;
		res.m21 = m21 * t;	res.m22 = m22 * t;	res.m23 = m23 * t;
		return res;
	}

	//! matrix scalar division
	inline __device__ __host__ float2x3 operator/(const float t) const
	{
		float2x3 res;
		res.m11 = m11 / t;	res.m12 = m12 / t;	res.m13 = m13 / t;
		res.m21 = m21 / t;	res.m22 = m22 / t;	res.m23 = m23 / t;
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

	inline __device__ __host__ const float* ptr() const {
		return entries;
	}
	inline __device__ __host__ float* ptr() {
		return entries;
	}

	inline __device__ __host__ float3x2 getTranspose()
	{
		float3x2 res;
		res.m11 = m11;
		res.m12 = m21;
		res.m21 = m12;
		res.m22 = m22;
		res.m31 = m13;
		res.m32 = m23;
		return res;
	}

	union
	{
		struct
		{
			float m11; float m12; float m13;
			float m21; float m22; float m23;
		};

		float entries[6];
		float entries2[2][3];
	};
};