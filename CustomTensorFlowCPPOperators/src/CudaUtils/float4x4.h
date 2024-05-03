#pragma once


#define MINF __int_as_float(0xff800000)
#define INF  __int_as_float(0x7f800000)

#include <iostream>
#include "cudaUtil.h"

#include "float3x3.h"
#include "float3x4.h"

#undef max
#undef min



class float4x4 {
public:
	inline __device__ __host__ float4x4() {

	}
	inline __device__ __host__ float4x4(const float values[16]) {
		m11 = values[0];	m12 = values[1];	m13 = values[2];	m14 = values[3];
		m21 = values[4];	m22 = values[5];	m23 = values[6];	m24 = values[7];
		m31 = values[8];	m32 = values[9];	m33 = values[10];	m34 = values[11];
		m41 = values[12];	m42 = values[13];	m43 = values[14];	m44 = values[15];
	}

	inline __device__ __host__ float4x4(const float4x4& other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
		m41 = other.m41;	m42 = other.m42;	m43 = other.m43;	m44 = other.m44;
	}

	//implicitly assumes last line to (0,0,0,1)
	inline __device__ __host__ float4x4(const float3x4& other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
		m41 = 0.0f;			m42 = 0.0f;			m43 = 0.0f;			m44 = 1.0f;
	}

	inline __device__ __host__ float4x4(const float3x3& other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = 0.0f;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = 0.0f;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = 0.0f;
		m41 = 0.0f;			m42 = 0.0f;			m43 = 0.0f;			m44 = 1.0f;
	}

	inline __device__ __host__ float4x4 operator=(const float4x4 &other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
		m41 = other.m41;	m42 = other.m42;	m43 = other.m43;	m44 = other.m44;
		return *this;
	}

	inline __device__ __host__ float4x4 operator=(const float3x4 &other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
		m41 = 0.0f;			m42 = 0.0f;			m43 = 0.0f;			m44 = 1.0f;
		return *this;
	}

	inline __device__ __host__ float4x4& operator=(const float3x3 &other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = 0.0f;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = 0.0f;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = 0.0f;
		m41 = 0.0f;			m42 = 0.0f;			m43 = 0.0f;			m44 = 1.0f;
		return *this;
	}


	//! not tested
	inline __device__ __host__ float4x4 operator*(const float4x4 &other) const {
		float4x4 res;
		res.m11 = m11*other.m11 + m12*other.m21 + m13*other.m31 + m14*other.m41;
		res.m12 = m11*other.m12 + m12*other.m22 + m13*other.m32 + m14*other.m42;
		res.m13 = m11*other.m13 + m12*other.m23 + m13*other.m33 + m14*other.m43;
		res.m14 = m11*other.m14 + m12*other.m24 + m13*other.m34 + m14*other.m44;

		res.m21 = m21*other.m11 + m22*other.m21 + m23*other.m31 + m24*other.m41;
		res.m22 = m21*other.m12 + m22*other.m22 + m23*other.m32 + m24*other.m42;
		res.m23 = m21*other.m13 + m22*other.m23 + m23*other.m33 + m24*other.m43;
		res.m24 = m21*other.m14 + m22*other.m24 + m23*other.m34 + m24*other.m44;

		res.m31 = m31*other.m11 + m32*other.m21 + m33*other.m31 + m34*other.m41;
		res.m32 = m31*other.m12 + m32*other.m22 + m33*other.m32 + m34*other.m42;
		res.m33 = m31*other.m13 + m32*other.m23 + m33*other.m33 + m34*other.m43;
		res.m34 = m31*other.m14 + m32*other.m24 + m33*other.m34 + m34*other.m44;

		res.m41 = m41*other.m11 + m42*other.m21 + m43*other.m31 + m44*other.m41;
		res.m42 = m41*other.m12 + m42*other.m22 + m43*other.m32 + m44*other.m42;
		res.m43 = m41*other.m13 + m42*other.m23 + m43*other.m33 + m44*other.m43;
		res.m44 = m41*other.m14 + m42*other.m24 + m43*other.m34 + m44*other.m44;

		return res;
	}

	// untested
	inline __device__ __host__ float4 operator*(const float4& v) const
	{
		return make_float4(
			m11*v.x + m12*v.y + m13*v.z + m14*v.w,
			m21*v.x + m22*v.y + m23*v.z + m24*v.w,
			m31*v.x + m32*v.y + m33*v.z + m34*v.w,
			m41*v.x + m42*v.y + m43*v.z + m44*v.w
			);
	}

	// untested
	//implicitly assumes w to be 1
	inline __device__ __host__ float3 operator*(const float3& v) const
	{
		return make_float3(
			m11*v.x + m12*v.y + m13*v.z + m14*1.0f,
			m21*v.x + m22*v.y + m23*v.z + m24*1.0f,
			m31*v.x + m32*v.y + m33*v.z + m34*1.0f
			);
	}

	inline __device__ __host__ float& operator()(int i, int j) {
		return entries2[i][j];
	}

	inline __device__ __host__ float operator()(int i, int j) const {
		return entries2[i][j];
	}


	static inline __device__ __host__  void swap(float& v0, float& v1) {
		float tmp = v0;
		v0 = v1;
		v1 = tmp;
	}

	inline __device__ __host__ void transpose() {
		swap(m12, m21);
		swap(m13, m31);
		swap(m23, m32);
		swap(m41, m14);
		swap(m42, m24);
		swap(m43, m34);
	}
	inline __device__ __host__ float4x4 getTranspose() const {
		float4x4 ret = *this;
		ret.transpose();
		return ret;
	}


	inline __device__ __host__ void invert() {
		*this = getInverse();
	}

	//! return the inverse matrix; but does not change the current matrix
	inline __device__ __host__ float4x4 getInverse() const {
		float inv[16];

		inv[0] = entries[5] * entries[10] * entries[15] -
			entries[5] * entries[11] * entries[14] -
			entries[9] * entries[6] * entries[15] +
			entries[9] * entries[7] * entries[14] +
			entries[13] * entries[6] * entries[11] -
			entries[13] * entries[7] * entries[10];

		inv[4] = -entries[4] * entries[10] * entries[15] +
			entries[4] * entries[11] * entries[14] +
			entries[8] * entries[6] * entries[15] -
			entries[8] * entries[7] * entries[14] -
			entries[12] * entries[6] * entries[11] +
			entries[12] * entries[7] * entries[10];

		inv[8] = entries[4] * entries[9] * entries[15] -
			entries[4] * entries[11] * entries[13] -
			entries[8] * entries[5] * entries[15] +
			entries[8] * entries[7] * entries[13] +
			entries[12] * entries[5] * entries[11] -
			entries[12] * entries[7] * entries[9];

		inv[12] = -entries[4] * entries[9] * entries[14] +
			entries[4] * entries[10] * entries[13] +
			entries[8] * entries[5] * entries[14] -
			entries[8] * entries[6] * entries[13] -
			entries[12] * entries[5] * entries[10] +
			entries[12] * entries[6] * entries[9];

		inv[1] = -entries[1] * entries[10] * entries[15] +
			entries[1] * entries[11] * entries[14] +
			entries[9] * entries[2] * entries[15] -
			entries[9] * entries[3] * entries[14] -
			entries[13] * entries[2] * entries[11] +
			entries[13] * entries[3] * entries[10];

		inv[5] = entries[0] * entries[10] * entries[15] -
			entries[0] * entries[11] * entries[14] -
			entries[8] * entries[2] * entries[15] +
			entries[8] * entries[3] * entries[14] +
			entries[12] * entries[2] * entries[11] -
			entries[12] * entries[3] * entries[10];

		inv[9] = -entries[0] * entries[9] * entries[15] +
			entries[0] * entries[11] * entries[13] +
			entries[8] * entries[1] * entries[15] -
			entries[8] * entries[3] * entries[13] -
			entries[12] * entries[1] * entries[11] +
			entries[12] * entries[3] * entries[9];

		inv[13] = entries[0] * entries[9] * entries[14] -
			entries[0] * entries[10] * entries[13] -
			entries[8] * entries[1] * entries[14] +
			entries[8] * entries[2] * entries[13] +
			entries[12] * entries[1] * entries[10] -
			entries[12] * entries[2] * entries[9];

		inv[2] = entries[1] * entries[6] * entries[15] -
			entries[1] * entries[7] * entries[14] -
			entries[5] * entries[2] * entries[15] +
			entries[5] * entries[3] * entries[14] +
			entries[13] * entries[2] * entries[7] -
			entries[13] * entries[3] * entries[6];

		inv[6] = -entries[0] * entries[6] * entries[15] +
			entries[0] * entries[7] * entries[14] +
			entries[4] * entries[2] * entries[15] -
			entries[4] * entries[3] * entries[14] -
			entries[12] * entries[2] * entries[7] +
			entries[12] * entries[3] * entries[6];

		inv[10] = entries[0] * entries[5] * entries[15] -
			entries[0] * entries[7] * entries[13] -
			entries[4] * entries[1] * entries[15] +
			entries[4] * entries[3] * entries[13] +
			entries[12] * entries[1] * entries[7] -
			entries[12] * entries[3] * entries[5];

		inv[14] = -entries[0] * entries[5] * entries[14] +
			entries[0] * entries[6] * entries[13] +
			entries[4] * entries[1] * entries[14] -
			entries[4] * entries[2] * entries[13] -
			entries[12] * entries[1] * entries[6] +
			entries[12] * entries[2] * entries[5];

		inv[3] = -entries[1] * entries[6] * entries[11] +
			entries[1] * entries[7] * entries[10] +
			entries[5] * entries[2] * entries[11] -
			entries[5] * entries[3] * entries[10] -
			entries[9] * entries[2] * entries[7] +
			entries[9] * entries[3] * entries[6];

		inv[7] = entries[0] * entries[6] * entries[11] -
			entries[0] * entries[7] * entries[10] -
			entries[4] * entries[2] * entries[11] +
			entries[4] * entries[3] * entries[10] +
			entries[8] * entries[2] * entries[7] -
			entries[8] * entries[3] * entries[6];

		inv[11] = -entries[0] * entries[5] * entries[11] +
			entries[0] * entries[7] * entries[9] +
			entries[4] * entries[1] * entries[11] -
			entries[4] * entries[3] * entries[9] -
			entries[8] * entries[1] * entries[7] +
			entries[8] * entries[3] * entries[5];

		inv[15] = entries[0] * entries[5] * entries[10] -
			entries[0] * entries[6] * entries[9] -
			entries[4] * entries[1] * entries[10] +
			entries[4] * entries[2] * entries[9] +
			entries[8] * entries[1] * entries[6] -
			entries[8] * entries[2] * entries[5];

		float matrixDet = entries[0] * inv[0] + entries[1] * inv[4] + entries[2] * inv[8] + entries[3] * inv[12];

		float matrixDetr = 1.0f / matrixDet;

		float4x4 res;
		for (unsigned int i = 0; i < 16; i++) {
			res.entries[i] = inv[i] * matrixDetr;
		}
		return res;

	}





	//! returns the 3x3 part of the matrix
	inline __device__ __host__ float3x3 getFloat3x3() {
		float3x3 ret;
		ret.m11 = m11;	ret.m12 = m12;	ret.m13 = m13;
		ret.m21 = m21;	ret.m22 = m22;	ret.m23 = m23;
		ret.m31 = m31;	ret.m32 = m32;	ret.m33 = m33;
		return ret;
	}

	//! sets the 3x3 part of the matrix (other values remain unchanged)
	inline __device__ __host__ void setFloat3x3(const float3x3 &other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;
	}

	//! sets the 4x4 part of the matrix to identity
	inline __device__ __host__ void setIdentity()
	{
		m11 = 1.0f;	m12 = 0.0f;	m13 = 0.0f;	m14 = 0.0f;
		m21 = 0.0f;	m22 = 1.0f;	m23 = 0.0f;	m24 = 0.0f;
		m31 = 0.0f;	m32 = 0.0f;	m33 = 1.0f;	m34 = 0.0f;
		m41 = 0.0f;	m42 = 0.0f;	m43 = 0.0f;	m44 = 1.0f;
	}

	//! sets the 4x4 part of the matrix to identity
	inline __device__ __host__ void setValue(float v)
	{
		m11 = v;	m12 = v;	m13 = v;	m14 = v;
		m21 = v;	m22 = v;	m23 = v;	m24 = v;
		m31 = v;	m32 = v;	m33 = v;	m34 = v;
		m41 = v;	m42 = v;	m43 = v;	m44 = v;
	}

	//! returns the 3x4 part of the matrix
	inline __device__ __host__ float3x4 getFloat3x4() {
		float3x4 ret;
		ret.m11 = m11;	ret.m12 = m12;	ret.m13 = m13;	ret.m14 = m14;
		ret.m21 = m21;	ret.m22 = m22;	ret.m23 = m23;	ret.m24 = m24;
		ret.m31 = m31;	ret.m32 = m32;	ret.m33 = m33;	ret.m34 = m34;
		return ret;
	}

	//! sets the 3x4 part of the matrix (other values remain unchanged)
	inline __device__ __host__ void setFloat3x4(const float3x4 &other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
	}




	inline __device__ __host__ const float* ptr() const {
		return entries;
	}
	inline __device__ __host__ float* ptr() {
		return entries;
	}

	union {
		struct {
			float m11; float m12; float m13; float m14;
			float m21; float m22; float m23; float m24;
			float m31; float m32; float m33; float m34;
			float m41; float m42; float m43; float m44;
		};
		float entries[16];
		float entries2[4][4];
	};
};