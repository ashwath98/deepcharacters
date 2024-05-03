//==============================================================================================//
// Classname:
//      ArapRotationHelper
//
//==============================================================================================//
// Description:
//      Computes the derivatives for ARAP constrain on the GPU
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#ifndef _ARAPROTATION_HELPER_
#define _ARAPROTATION_HELPER_

//==============================================================================================//

#include "cuda_SimpleMatrixUtil.h"

//==============================================================================================//

// Rotation times vector
inline __device__ mat3x3 arapEvalDerivativeRotationTimesVector(const float3x3& dRAlpha, const float3x3& dRBeta, const float3x3& dRGamma, const float3& d)
{
	mat3x3 R; 
	float3 b = dRAlpha*d; 
	R(0, 0) = b.x; R(1, 0) = b.y; R(2, 0) = b.z;
	
	b = dRBeta *d;
	R(0, 1) = b.x; R(1, 1) = b.y; R(2, 1) = b.z;
	
	b = dRGamma*d; 
	R(0, 2) = b.x; R(1, 2) = b.y; R(2, 2) = b.z;

	return R;
}

//==============================================================================================//

// Rotation Matrix
inline __device__ mat3x3 arapEvalRMat(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma)
{
	mat3x3 R;
	R(0, 0) = CosGamma*CosBeta;
	R(0, 1) = -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha;
	R(0, 2) = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;

	R(1, 0) = SinGamma*CosBeta;
	R(1, 1) = CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha;
	R(1, 2) = -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha;

	R(2, 0) = -SinBeta;
	R(2, 1) = CosBeta*SinAlpha;
	R(2, 2) = CosBeta*CosAlpha;

	return R;
}

//==============================================================================================//

inline __device__ mat3x3 arapEvalRMat(const float3& angles)
{
	const float cosAlpha = cos(angles.x); float cosBeta = cos(angles.y); float cosGamma = cos(angles.z);
	const float sinAlpha = sin(angles.x); float sinBeta = sin(angles.y); float sinGamma = sin(angles.z);

	return arapEvalRMat(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
}

//==============================================================================================//

// Rotation Matrix
inline __device__ float3 arapEvalRAngles(const float3x3& R)
{
	const float PI = 3.14159265359f;
	if (fabs(R.m31) != 1.0f)
	{
		const float beta = -asin(R.m31); 
		const float cosBeta = cos(beta);

		return make_float3(atan2(R.m32, R.m33), beta, atan2(R.m21, R.m11));

		// Solution not unique, this is the second one
		//const float beta = PI+asin(R.m31); const float cosBeta = cos(beta);
		//return make_float3(atan2(R.m32/cosBeta, R.m33/cosBeta), beta, atan2(R.m21/cosBeta, R.m11/cosBeta));
	}
	else
	{
		if (R.m31 == -1.0f)
		{
			return make_float3(atan2(R.m12, R.m13), PI / 2, 0.0f);
		}
			
		else
		{
			return make_float3(atan2(-R.m12, -R.m13), -PI / 2, 0.0f);
		}
	}
}

//==============================================================================================//

inline __device__ mat3x3 arapEvalR(const float3& angles) // angles = [alpha, beta, gamma]
{
	return arapEvalRMat(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z));
}

//==============================================================================================//

// Rotation Matrix dAlpha
inline __device__ mat3x3 arapEvalRMat_dAlpha(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma)
{
	mat3x3 R;
	R(0, 0) = 0.0f;
	R(0, 1) = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;
	R(0, 2) = SinGamma*CosAlpha - CosGamma*SinBeta*SinAlpha;

	R(1, 0) = 0.0f;
	R(1, 1) = -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha;
	R(1, 2) = -CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha;

	R(2, 0) = 0.0f;
	R(2, 1) = CosBeta*CosAlpha;
	R(2, 2) = -CosBeta*SinAlpha;

	return R;
}

//==============================================================================================//

inline __device__ mat3x3 arapEvalR_dAlpha(float3 angles) // angles = [alpha, beta, gamma]
{
	return arapEvalRMat_dAlpha(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z));
}

//==============================================================================================//

// Rotation Matrix dBeta
inline __device__ mat3x3 arapEvalRMat_dBeta(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma)
{
	mat3x3 R;
	R(0, 0) = -CosGamma*SinBeta;
	R(0, 1) = CosGamma*CosBeta*SinAlpha;
	R(0, 2) = CosGamma*CosBeta*CosAlpha;

	R(1, 0) = -SinGamma*SinBeta;
	R(1, 1) = SinGamma*CosBeta*SinAlpha;
	R(1, 2) = SinGamma*CosBeta*CosAlpha;

	R(2, 0) = -CosBeta;
	R(2, 1) = -SinBeta*SinAlpha;
	R(2, 2) = -SinBeta*CosAlpha;

	return R;
}

//==============================================================================================//

inline __device__ mat3x3 arapEvalR_dBeta(float3 angles) // angles = [alpha, beta, gamma]
{
	return arapEvalRMat_dBeta(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z));
}

//==============================================================================================//

// Rotation Matrix dGamma
inline __device__ mat3x3 arapEvalRMat_dGamma(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma)
{
	mat3x3 R;
	R(0, 0) = -SinGamma*CosBeta;
	R(0, 1) = -CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha;
	R(0, 2) = CosGamma*SinAlpha - SinGamma*SinBeta*CosAlpha;

	R(1, 0) = CosGamma*CosBeta;
	R(1, 1) = -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha;
	R(1, 2) = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;

	R(2, 0) = 0.0f;
	R(2, 1) = 0.0f;
	R(2, 2) = 0.0f;

	return R;
}

//==============================================================================================//

inline __device__ mat3x3 arapEvalR_dGamma(float3 angles) // angles = [alpha, beta, gamma]
{
	return arapEvalRMat_dGamma(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z));
}

//==============================================================================================//

// Rotation Matrix dIdx
inline __device__ mat3x3 arapEvalR_dIdx(float3 angles, unsigned int idx) // 0 = alpha, 1 = beta, 2 = gamma
{
	if (idx == 0) return arapEvalR_dAlpha(angles);
	else if (idx == 1) return arapEvalR_dBeta(angles);
	else return arapEvalR_dGamma(angles);
}

//==============================================================================================//

inline __device__ void arapEvalDerivativeRotationMatrix(const float3& angles, mat3x3& dRAlpha, mat3x3& dRBeta, mat3x3& dRGamma)
{
	const float cosAlpha = cos(angles.x); float cosBeta = cos(angles.y); float cosGamma = cos(angles.z);
	const float sinAlpha = sin(angles.x); float sinBeta = sin(angles.y); float sinGamma = sin(angles.z);

	dRAlpha = arapEvalRMat_dAlpha(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
	dRBeta = arapEvalRMat_dBeta(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
	dRGamma = arapEvalRMat_dGamma(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
}

//==============================================================================================//

#endif