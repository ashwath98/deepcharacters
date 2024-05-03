#pragma once


#define MINF __int_as_float(0xff800000)
#define INF  __int_as_float(0x7f800000)

#include <iostream>
#include "cudaUtil.h"

#include "float2x2.h"
#include "float3x2.h"

#undef max
#undef min





class float3x3 {
public:
	inline __device__ __host__ float3x3() {

	}
	inline __device__ __host__ float3x3(const float values[9]) {
		m11 = values[0];	m12 = values[1];	m13 = values[2];
		m21 = values[3];	m22 = values[4];	m23 = values[5];
		m31 = values[6];	m32 = values[7];	m33 = values[8];
	}

	inline __device__ __host__ float3x3(const float3x3& other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;
	}

	explicit inline __device__ __host__ float3x3(const float2x2& other) {
		m11 = other.m11;	m12 = other.m12;	m13 = 0.0;
		m21 = other.m21;	m22 = other.m22;	m23 = 0.0;
		m31 = 0.0;			m32 = 0.0;			m33 = 0.0;
	}

	inline __device__ __host__ float3x3& operator=(const float3x3 &other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;
		return *this;
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
	}
	inline __device__ __host__ float3x3 getTranspose() const {
		float3x3 ret = *this;
		ret.transpose();
		return ret;
	}

	//! inverts the matrix
	inline __device__ __host__ void invert() {
		*this = getInverse();
	}

	//! computes the inverse of the matrix; the result is returned
	inline __device__ __host__ float3x3 getInverse() const {
		float3x3 res;
		res.entries[0] = entries[4] * entries[8] - entries[5] * entries[7];
		res.entries[1] = -entries[1] * entries[8] + entries[2] * entries[7];
		res.entries[2] = entries[1] * entries[5] - entries[2] * entries[4];

		res.entries[3] = -entries[3] * entries[8] + entries[5] * entries[6];
		res.entries[4] = entries[0] * entries[8] - entries[2] * entries[6];
		res.entries[5] = -entries[0] * entries[5] + entries[2] * entries[3];

		res.entries[6] = entries[3] * entries[7] - entries[4] * entries[6];
		res.entries[7] = -entries[0] * entries[7] + entries[1] * entries[6];
		res.entries[8] = entries[0] * entries[4] - entries[1] * entries[3];
		float nom = 1.0f / det();
		return res * nom;
	}

	inline __device__ __host__ void setZero(float value = 0.0f) {
		m11 = m12 = m13 = value;
		m21 = m22 = m23 = value;
		m31 = m32 = m33 = value;
	}

	inline __device__ __host__ float det() const {
		return
			+m11*m22*m33
			+ m12*m23*m31
			+ m13*m21*m32
			- m31*m22*m13
			- m32*m23*m11
			- m33*m21*m12;
	}

	inline __device__ __host__ float3 getRow(unsigned int i) {
		return make_float3(entries[3 * i + 0], entries[3 * i + 1], entries[3 * i + 2]);
	}

	inline __device__ __host__ void setRow(unsigned int i, float3& r) {
		entries[3 * i + 0] = r.x;
		entries[3 * i + 1] = r.y;
		entries[3 * i + 2] = r.z;
	}
	inline __device__ __host__ void setAll(float value) {
		m11 = m12 = m13 = value;
		m21 = m22 = m23 = value;
		m31 = m32 = m33 = value;
	}

	inline __device__ __host__ void normalizeRows()
	{
		//#pragma unroll 3
		for (unsigned int i = 0; i<3; i++)
		{
			float3 r = getRow(i);
			r /= length(r);
			setRow(i, r);
		}
	}

	//! computes the product of two matrices (result stored in this)
	inline __device__ __host__ void mult(const float3x3 &other) {
		float3x3 res;
		res.m11 = m11 * other.m11 + m12 * other.m21 + m13 * other.m31;
		res.m12 = m11 * other.m12 + m12 * other.m22 + m13 * other.m32;
		res.m13 = m11 * other.m13 + m12 * other.m23 + m13 * other.m33;

		res.m21 = m21 * other.m11 + m22 * other.m21 + m23 * other.m31;
		res.m22 = m21 * other.m12 + m22 * other.m22 + m23 * other.m32;
		res.m23 = m21 * other.m13 + m22 * other.m23 + m23 * other.m33;

		res.m31 = m21 * other.m11 + m32 * other.m21 + m33 * other.m31;
		res.m32 = m21 * other.m12 + m32 * other.m22 + m33 * other.m32;
		res.m33 = m21 * other.m13 + m32 * other.m23 + m33 * other.m33;
		*this = res;
	}

	//! computes the sum of two matrices (result stored in this)
	inline __device__ __host__ void add(const float3x3 &other) {
		m11 += other.m11;	m12 += other.m12;	m13 += other.m13;
		m21 += other.m21;	m22 += other.m22;	m23 += other.m23;
		m31 += other.m31;	m32 += other.m32;	m33 += other.m33;
	}

	//! standard matrix matrix multiplication
	inline __device__ __host__ float3x3 operator*(const float3x3 &other) const {
		float3x3 res;
		res.m11 = m11 * other.m11 + m12 * other.m21 + m13 * other.m31;
		res.m12 = m11 * other.m12 + m12 * other.m22 + m13 * other.m32;
		res.m13 = m11 * other.m13 + m12 * other.m23 + m13 * other.m33;

		res.m21 = m21 * other.m11 + m22 * other.m21 + m23 * other.m31;
		res.m22 = m21 * other.m12 + m22 * other.m22 + m23 * other.m32;
		res.m23 = m21 * other.m13 + m22 * other.m23 + m23 * other.m33;

		res.m31 = m31 * other.m11 + m32 * other.m21 + m33 * other.m31;
		res.m32 = m31 * other.m12 + m32 * other.m22 + m33 * other.m32;
		res.m33 = m31 * other.m13 + m32 * other.m23 + m33 * other.m33;
		return res;
	}

	//! standard matrix matrix multiplication
	inline __device__ __host__ float3x2 operator*(const float3x2 &other) const {
		float3x2 res;
		res.m11 = m11 * other.m11 + m12 * other.m21 + m13 * other.m31;
		res.m12 = m11 * other.m12 + m12 * other.m22 + m13 * other.m32;

		res.m21 = m21 * other.m11 + m22 * other.m21 + m23 * other.m31;
		res.m22 = m21 * other.m12 + m22 * other.m22 + m23 * other.m32;

		res.m31 = m31 * other.m11 + m32 * other.m21 + m33 * other.m31;
		res.m32 = m31 * other.m12 + m32 * other.m22 + m33 * other.m32;
		return res;
	}

	inline __device__ __host__ float3 operator*(const float3 &v) const {
		return make_float3(
			m11*v.x + m12*v.y + m13*v.z,
			m21*v.x + m22*v.y + m23*v.z,
			m31*v.x + m32*v.y + m33*v.z
			);
	}

	inline __device__ __host__ float3x3 operator*(const float t) const {
		float3x3 res;
		res.m11 = m11 * t;		res.m12 = m12 * t;		res.m13 = m13 * t;
		res.m21 = m21 * t;		res.m22 = m22 * t;		res.m23 = m23 * t;
		res.m31 = m31 * t;		res.m32 = m32 * t;		res.m33 = m33 * t;
		return res;
	}


	inline __device__ __host__ float3x3 operator+(const float3x3 &other) const {
		float3x3 res;
		res.m11 = m11 + other.m11;	res.m12 = m12 + other.m12;	res.m13 = m13 + other.m13;
		res.m21 = m21 + other.m21;	res.m22 = m22 + other.m22;	res.m23 = m23 + other.m23;
		res.m31 = m31 + other.m31;	res.m32 = m32 + other.m32;	res.m33 = m33 + other.m33;
		return res;
	}

	inline __device__ __host__ float3x3 operator-(const float3x3 &other) const {
		float3x3 res;
		res.m11 = m11 - other.m11;	res.m12 = m12 - other.m12;	res.m13 = m13 - other.m13;
		res.m21 = m21 - other.m21;	res.m22 = m22 - other.m22;	res.m23 = m23 - other.m23;
		res.m31 = m31 - other.m31;	res.m32 = m32 - other.m32;	res.m33 = m33 - other.m33;
		return res;
	}

	static inline __device__ __host__ float3x3 getIdentity() {
		float3x3 res;
		res.setZero();
		res.m11 = res.m22 = res.m33 = 1.0f;
		return res;
	}

	static inline __device__ __host__ float3x3 getZeroMatrix() {
		float3x3 res;
		res.setZero();
		return res;
	}

	static inline __device__ __host__ float3x3 getDiagonalMatrix(float diag = 1.0f) {
		float3x3 res;
		res.m11 = diag;		res.m12 = 0.0f;		res.m13 = 0.0f;
		res.m21 = 0.0f;		res.m22 = diag;		res.m23 = 0.0f;
		res.m31 = 0.0f;		res.m32 = 0.0f;		res.m33 = diag;
		return res;
	}

	static inline __device__ __host__  float3x3 tensorProduct(const float3 &v, const float3 &vt) {
		float3x3 res;
		res.m11 = v.x * vt.x;	res.m12 = v.x * vt.y;	res.m13 = v.x * vt.z;
		res.m21 = v.y * vt.x;	res.m22 = v.y * vt.y;	res.m23 = v.y * vt.z;
		res.m31 = v.z * vt.x;	res.m32 = v.z * vt.y;	res.m33 = v.z * vt.z;
		return res;
	}

	inline __device__ __host__ const float* ptr() const {
		return entries;
	}
	inline __device__ __host__ float* ptr() {
		return entries;
	}

	union {
		struct {
			float m11; float m12; float m13;
			float m21; float m22; float m23;
			float m31; float m32; float m33;
		};
		float entries[9];
		float entries2[3][3];
	};
};
