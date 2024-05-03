#pragma once

#define MINF __int_as_float(0xff800000)
#define INF  __int_as_float(0x7f800000)

#include <iostream>
#include "cudaUtil.h"
#undef max
#undef min


//////////////////////////////
// float2x2
//////////////////////////////

class float2x2
{
	public:

		inline __device__ __host__ float2x2()
		{
		}

		inline __device__ __host__ float2x2(const float values[4])
		{
			m11 = values[0];	m12 = values[1];
			m21 = values[2];	m22 = values[3];
		}

		inline __device__ __host__ float2x2(const float2x2& other)
		{
			m11 = other.m11;	m12 = other.m12;
			m21 = other.m21; 	m22 = other.m22;
		}

		inline __device__ __host__ void setZero()
		{
			m11 = 0.0f;	m12 = 0.0f;
			m21 = 0.0f; m22 = 0.0f;
		}

		static inline __device__ __host__ float2x2 getIdentity()
		{
			float2x2 res;
			res.setZero();
			res.m11 = res.m22 = 1.0f;
			return res;
		}

		inline __device__ __host__ float2x2& operator=(const float2x2 &other)
		{
			m11 = other.m11;	m12 = other.m12;
			m21 = other.m21;	m22 = other.m22;
			return *this;
		}

		inline __device__ __host__ float2x2 getInverse()
		{
			float2x2 res;
			res.m11 = m22; res.m12 = -m12;
			res.m21 = -m21; res.m22 = m11;

			return res*(1.0f / det());
		}

		inline __device__ __host__ float det()
		{
			return m11*m22 - m21*m12;
		}

		inline __device__ __host__ float2 operator*(const float2& v) const
		{
			return make_float2(m11*v.x + m12*v.y, m21*v.x + m22*v.y);
		}

		//! matrix scalar multiplication
		inline __device__ __host__ float2x2 operator*(const float t) const
		{
			float2x2 res;
			res.m11 = m11 * t;	res.m12 = m12 * t;
			res.m21 = m21 * t;	res.m22 = m22 * t;
			return res;
		}

		//! matrix matrix multiplication
		inline __device__ __host__ float2x2 operator*(const float2x2& other) const
		{
			float2x2 res;
			res.m11 = m11 * other.m11 + m12 * other.m21;
			res.m12 = m11 * other.m12 + m12 * other.m22;
			res.m21 = m21 * other.m11 + m22 * other.m21;
			res.m22 = m21 * other.m12 + m22 * other.m22;
			return res;
		}

		//! matrix matrix addition
		inline __device__ __host__ float2x2 operator+(const float2x2& other) const
		{
			float2x2 res;
			res.m11 = m11 + other.m11;
			res.m12 = m12 + other.m12;
			res.m21 = m21 + other.m21;
			res.m22 = m22 + other.m22;
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

		union
		{
			struct
			{
				float m11; float m12;
				float m21; float m22;
			};

			float entries[4];
			float entries2[2][2];
		};
};