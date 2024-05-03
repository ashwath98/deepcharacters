#pragma once


#define MINF __int_as_float(0xff800000)
#define INF  __int_as_float(0x7f800000)

#include <iostream>
#include "cudaUtil.h"
#include "float3x3.h"

#undef max
#undef min



class float3x4 
{
public:
	inline __device__ __host__ float3x4() {

	}
	inline __device__ __host__ float3x4(const float values[12]) {
		m11 = values[0];	m12 = values[1];	m13 = values[2];	m14 = values[3];
		m21 = values[4];	m22 = values[5];	m23 = values[6];	m24 = values[7];
		m31 = values[8];	m32 = values[9];	m33 = values[10];	m34 = values[11];
	}

	inline __device__ __host__ float3x4(const float3x4& other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
	}

	inline __device__ __host__ float3x4(const float3x3& other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = 0.0f;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = 0.0f;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = 0.0f;
	}

	inline __device__ __host__ float3x4 operator=(const float3x4 &other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = other.m14;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = other.m24;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = other.m34;
		return *this;
	}

	inline __device__ __host__ float3x4& operator=(const float3x3 &other) {
		m11 = other.m11;	m12 = other.m12;	m13 = other.m13;	m14 = 0.0f;
		m21 = other.m21;	m22 = other.m22;	m23 = other.m23;	m24 = 0.0f;
		m31 = other.m31;	m32 = other.m32;	m33 = other.m33;	m34 = 0.0f;
		return *this;
	}

	//! assumes the last line of the matrix implicitly to be (0,0,0,1)
	inline __device__ __host__ float3 operator*(const float4 &v) const {
		return make_float3(
			m11*v.x + m12*v.y + m13*v.z + m14*v.w,
			m21*v.x + m22*v.y + m23*v.z + m24*v.w,
			m31*v.x + m32*v.y + m33*v.z + m34*v.w
			);
	}

	//! assumes an implicit 1 in w component of the input vector
	inline __device__ __host__ float3 operator*(const float3 &v) const {
		return make_float3(
			m11*v.x + m12*v.y + m13*v.z + m14,
			m21*v.x + m22*v.y + m23*v.z + m24,
			m31*v.x + m32*v.y + m33*v.z + m34
			);
	}

	//! matrix scalar multiplication
	inline __device__ __host__ float3x4 operator*(const float t) const {
		float3x4 res;
		res.m11 = m11 * t;		res.m12 = m12 * t;		res.m13 = m13 * t;		res.m14 = m14 * t;
		res.m21 = m21 * t;		res.m22 = m22 * t;		res.m23 = m23 * t;		res.m24 = m24 * t;
		res.m31 = m31 * t;		res.m32 = m32 * t;		res.m33 = m33 * t;		res.m34 = m34 * t;
		return res;
	}
	inline __device__ __host__ float3x4& operator*=(const float t) {
		*this = *this * t;
		return *this;
	}

	//! matrix scalar division
	inline __device__ __host__ float3x4 operator/(const float t) const {
		float3x4 res;
		res.m11 = m11 / t;		res.m12 = m12 / t;		res.m13 = m13 / t;		res.m14 = m14 / t;
		res.m21 = m21 / t;		res.m22 = m22 / t;		res.m23 = m23 / t;		res.m24 = m24 / t;
		res.m31 = m31 / t;		res.m32 = m32 / t;		res.m33 = m33 / t;		res.m34 = m34 / t;
		return res;
	}
	inline __device__ __host__ float3x4& operator/=(const float t) {
		*this = *this / t;
		return *this;
	}

	//! assumes the last line of the matrix implicitly to be (0,0,0,1)
	inline __device__ __host__ float3x4 operator*(const float3x4 &other) const {
		float3x4 res;
		res.m11 = m11*other.m11 + m12*other.m21 + m13*other.m31;
		res.m12 = m11*other.m12 + m12*other.m22 + m13*other.m32;
		res.m13 = m11*other.m13 + m12*other.m23 + m13*other.m33;
		res.m14 = m11*other.m14 + m12*other.m24 + m13*other.m34 + m14;

		res.m21 = m21*other.m11 + m22*other.m21 + m23*other.m31;
		res.m22 = m21*other.m12 + m22*other.m22 + m23*other.m32;
		res.m23 = m21*other.m13 + m22*other.m23 + m23*other.m33;
		res.m24 = m21*other.m14 + m22*other.m24 + m23*other.m34 + m24;

		res.m31 = m31*other.m11 + m32*other.m21 + m33*other.m31;
		res.m32 = m31*other.m12 + m32*other.m22 + m33*other.m32;
		res.m33 = m31*other.m13 + m32*other.m23 + m33*other.m33;
		res.m34 = m31*other.m14 + m32*other.m24 + m33*other.m34 + m34;

		//res.m41 = m41*other.m11 + m42*other.m21 + m43*other.m31 + m44*other.m41;  
		//res.m42 = m41*other.m12 + m42*other.m22 + m43*other.m32 + m44*other.m42;  
		//res.m43 = m41*other.m13 + m42*other.m23 + m43*other.m33 + m44*other.m43; 
		//res.m44 = m41*other.m14 + m42*other.m24 + m43*other.m34 + m44*other.m44;

		return res;
	}

	//! assumes the last line of the matrix implicitly to be (0,0,0,1); and a (0,0,0) translation of other
	inline __device__ __host__ float3x4 operator*(const float3x3 &other) const {
		float3x4 res;
		res.m11 = m11*other.m11 + m12*other.m21 + m13*other.m31;
		res.m12 = m11*other.m12 + m12*other.m22 + m13*other.m32;
		res.m13 = m11*other.m13 + m12*other.m23 + m13*other.m33;
		res.m14 = m14;

		res.m21 = m21*other.m11 + m22*other.m21 + m23*other.m31;
		res.m22 = m21*other.m12 + m22*other.m22 + m23*other.m32;
		res.m23 = m21*other.m13 + m22*other.m23 + m23*other.m33;
		res.m24 = m24;

		res.m31 = m31*other.m11 + m32*other.m21 + m33*other.m31;
		res.m32 = m31*other.m12 + m32*other.m22 + m33*other.m32;
		res.m33 = m31*other.m13 + m32*other.m23 + m33*other.m33;
		res.m34 = m34;

		return res;
	}



	inline __device__ __host__ float& operator()(int i, int j) {
		return entries2[i][j];
	}

	inline __device__ __host__ float operator()(int i, int j) const {
		return entries2[i][j];
	}

	//! returns the translation part of the matrix
	inline __device__ __host__ float3 getTranslation() {
		return make_float3(m14, m24, m34);
	}

	//! sets only the translation part of the matrix (other values remain unchanged)
	inline __device__ __host__ void setTranslation(const float3 &t) {
		m14 = t.x;
		m24 = t.y;
		m34 = t.z;
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

	//! inverts the matrix
	inline __device__ __host__ void inverse() {
		*this = getInverse();
	}

	//! computes the inverse of the matrix
	inline __device__ __host__ float3x4 getInverse() {
		float3x3 A = getFloat3x3();
		A.invert();
		float3 t = getTranslation();
		t = A*t;

		float3x4 ret;
		ret.setFloat3x3(A);
		ret.setTranslation(make_float3(-t.x, -t.y, -t.z));	//float3 doesn't have unary '-'... thank you cuda
		return ret;
	}

	//! prints the matrix; only host	
	__host__ void print() {
		std::cout <<
			m11 << " " << m12 << " " << m13 << " " << m14 << std::endl <<
			m21 << " " << m22 << " " << m23 << " " << m24 << std::endl <<
			m31 << " " << m32 << " " << m33 << " " << m34 << std::endl <<
			std::endl;
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
		};
		float entries[9];
		float entries2[3][4];
	};
};
