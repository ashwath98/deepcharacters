//==============================================================================================//
// Classname:
//      DQHelper
//
//==============================================================================================//
// Description:
//      Some useful functions for dual quaternion conversions and gradient calculations
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include "cuda_SimpleMatrixUtil.h"

//==============================================================================================//
//functions
//==============================================================================================//

//convert dq to rotation matrix
inline __device__ float3x3 dq2RotMatrix(const float4& dqRot)
{
	float3x3 R;

	//rotation
	float twx = 2.f * dqRot.x * dqRot.w;
	float twy = 2.f * dqRot.y * dqRot.w;
	float twz = 2.f * dqRot.z * dqRot.w;
	float txx = 2.f * dqRot.x * dqRot.x;
	float txy = 2.f * dqRot.y * dqRot.x;
	float txz = 2.f * dqRot.z * dqRot.x;
	float tyy = 2.f * dqRot.y * dqRot.y;
	float tyz = 2.f * dqRot.z * dqRot.y;
	float tzz = 2.f * dqRot.z * dqRot.z;

	R(0, 0) = 1.f - tyy - tzz;
	R(0, 1) = txy - twz;
	R(0, 2) = txz + twy;
	R(1, 0) = txy + twz;
	R(1, 1) = 1.f - txx - tzz;
	R(1, 2) = tyz - twx;
	R(2, 0) = txz - twy;
	R(2, 1) = tyz + twx;
	R(2, 2) = 1.f - txx - tyy;

	return R;
}

//==============================================================================================//

//apply rotation quaternion to point
inline __device__ float3 dq2RotatedPoint(const float4& dqRot, const float3& vert)
{
	float R00 = 1.f - 2.f * dqRot.y * dqRot.y - 2.f * dqRot.z * dqRot.z;
	float R01 = 2.f * dqRot.y * dqRot.x - 2.f * dqRot.z * dqRot.w;
	float R02 = 2.f * dqRot.z * dqRot.x + 2.f * dqRot.y * dqRot.w;

	float R10 = 2.f * dqRot.y * dqRot.x + 2.f * dqRot.z * dqRot.w;
	float R11 = 1.f - 2.f * dqRot.x * dqRot.x - 2.f * dqRot.z * dqRot.z;
	float R12 = 2.f * dqRot.z * dqRot.y - 2.f * dqRot.x * dqRot.w;

	float R20 = 2.f * dqRot.z * dqRot.x - 2.f * dqRot.y * dqRot.w;
	float R21 = 2.f * dqRot.z * dqRot.y + 2.f * dqRot.x * dqRot.w;
	float R22 = 1.f - 2.f * dqRot.x * dqRot.x - 2.f * dqRot.y * dqRot.y;

	float v0 = R00 * vert.x + R01 * vert.y + R02 * vert.z;
	float v1 = R10 * vert.x + R11 * vert.y + R12 * vert.z;
	float v2 = R20 * vert.x + R21 * vert.y + R22 * vert.z;

	return make_float3(v0, v1, v2);
}

//==============================================================================================//

//convert dq to translation vector
inline __device__ float3 dq2TransVector(const float4& dqRot, const float4& dqTrans)
{
	float3 t = make_float3(0.f, 0.f, 0.f);

	t.x = 2.0f * (-dqTrans.w * dqRot.x + dqTrans.x * dqRot.w - dqTrans.y * dqRot.z + dqTrans.z * dqRot.y);
	t.y = 2.0f * (-dqTrans.w * dqRot.y + dqTrans.x * dqRot.z + dqTrans.y * dqRot.w - dqTrans.z * dqRot.x);
	t.z = 2.0f * (-dqTrans.w * dqRot.z - dqTrans.x * dqRot.y + dqTrans.y * dqRot.x + dqTrans.z * dqRot.w);

	return t;
}

//==============================================================================================//
//Jacobi
//==============================================================================================//

//grad of convert dq to rotation matrix
inline __device__ float3x4 dq2RotatedPointJacobiDQ(const float4& dqRot, const float3& vert)
{
	float3x4 jacobi_RotV_dq;
	float x = dqRot.x;
	float y = dqRot.y;
	float z = dqRot.z;
	float w = dqRot.w;

	// ====================================================================================
	
	//float R00 = 1.f - 2.f * dqRot.y * dqRot.y - 2.f * dqRot.z * dqRot.z;
	float R00_dq0 = 0.f;
	float R00_dq1 = -4.f * y;
	float R00_dq2 = -4.f * z;
	float R00_dq3 = 0.f;

	//float R01 = 2.f * dqRot.y * dqRot.x - 2.f * dqRot.z * dqRot.w;
	float R01_dq0 =   2.f * y;
	float R01_dq1 =   2.f * x;
	float R01_dq2 = - 2.f * w;
	float R01_dq3 = - 2.f * z;

	//float R02 = 2.f * dqRot.z * dqRot.x + 2.f * dqRot.y * dqRot.w;
	float R02_dq0 = 2.f * z;
	float R02_dq1 = 2.f * w;
	float R02_dq2 = 2.f * x;
	float R02_dq3 = 2.f * y;

	jacobi_RotV_dq(0, 0) = R00_dq0 * vert.x + R01_dq0 *  vert.y + R02_dq0 + vert.z;
	jacobi_RotV_dq(0, 1) = R00_dq1 * vert.x + R01_dq1 *  vert.y + R02_dq1 + vert.z;
	jacobi_RotV_dq(0, 2) = R00_dq2 * vert.x + R01_dq2 *  vert.y + R02_dq2 + vert.z;
	jacobi_RotV_dq(0, 3) = R00_dq3 * vert.x + R01_dq3 *  vert.y + R02_dq3 + vert.z;

	// ====================================================================================
	
	//float R10 = 2.f * dqRot.y * dqRot.x + 2.f * dqRot.z * dqRot.w;
	float R10_dq0 = 2.f * y;
	float R10_dq1 = 2.f * x;
	float R10_dq2 = 2.f * w;
	float R10_dq3 = 2.f * z;

	//float R11 = 1.f - 2.f * dqRot.x * dqRot.x - 2.f * dqRot.z * dqRot.z;
	float R11_dq0 = -4.f * x;
	float R11_dq1 = 0.f;
	float R11_dq2 = -4.f * z;
	float R11_dq3 = 0.f;

	//float R12 = 2.f * dqRot.z * dqRot.y - 2.f * dqRot.x * dqRot.w;
	float R12_dq0 = - 2.f * w;
	float R12_dq1 =   2.f * z;
	float R12_dq2 =   2.f * y;
	float R12_dq3 = - 2.f * x;

	jacobi_RotV_dq(1, 0) = R10_dq0 *  vert.x + R11_dq0 *  vert.y + R12_dq0 + vert.z;
	jacobi_RotV_dq(1, 1) = R10_dq1 *  vert.x + R11_dq1 *  vert.y + R12_dq1 + vert.z;
	jacobi_RotV_dq(1, 2) = R10_dq2 *  vert.x + R11_dq2 *  vert.y + R12_dq2 + vert.z;
	jacobi_RotV_dq(1, 3) = R10_dq3 *  vert.x + R11_dq3 *  vert.y + R12_dq3 + vert.z;

	// ====================================================================================

	//float R20 = 2.f * dqRot.z * dqRot.x - 2.f * dqRot.y * dqRot.w;
	float R20_dq0 =   2.f * z;
	float R20_dq1 = - 2.f * w;
	float R20_dq2 =   2.f * x;
	float R20_dq3 = - 2.f * y;

	//float R21 = 2.f * dqRot.z * dqRot.y + 2.f * dqRot.x * dqRot.w;
	float R21_dq0 = 2.f * w;
	float R21_dq1 = 2.f * z;
	float R21_dq2 = 2.f * y;
	float R21_dq3 = 2.f * x;

	//float R22 = 1.f - 2.f * dqRot.x * dqRot.x - 2.f * dqRot.y * dqRot.y;
	float R22_dq0 = - 4.f * x;
	float R22_dq1 = - 4.f * y;
	float R22_dq2 = 0.f;
	float R22_dq3 = 0.f;

	jacobi_RotV_dq(2, 0) = R20_dq0 *  vert.x + R21_dq0 *  vert.y + R22_dq0 + vert.z;
	jacobi_RotV_dq(2, 1) = R20_dq1 *  vert.x + R21_dq1 *  vert.y + R22_dq1 + vert.z;
	jacobi_RotV_dq(2, 2) = R20_dq2 *  vert.x + R21_dq2 *  vert.y + R22_dq2 + vert.z;
	jacobi_RotV_dq(2, 3) = R20_dq3 *  vert.x + R21_dq3 *  vert.y + R22_dq3 + vert.z;

	// ====================================================================================

	return jacobi_RotV_dq;
}

//==============================================================================================//

//convert dq to translation vector
inline __device__ float3x4 dq2TransVectorJacobiDQRot(const float4& dqRot, const float4& dqTrans)
{
	float3x4 jacobi_RotV_dq;

	//int i = 2.0f * (-dqTrans.w * dqRot.x + dqTrans.x * dqRot.w - dqTrans.y * dqRot.z + dqTrans.z * dqRot.y);
	jacobi_RotV_dq(0, 0) = 2.0f * (- dqTrans.w);
	jacobi_RotV_dq(0, 1) = 2.0f * (  dqTrans.z);
	jacobi_RotV_dq(0, 2) = 2.0f * (- dqTrans.y);
	jacobi_RotV_dq(0, 3) = 2.0f * (  dqTrans.x);


	//int i = 2.0f * (-dqTrans.w * dqRot.y + dqTrans.x * dqRot.z + dqTrans.y * dqRot.w - dqTrans.z * dqRot.x);
	jacobi_RotV_dq(1, 0) = 2.0f * (- dqTrans.z);
	jacobi_RotV_dq(1, 1) = 2.0f * (- dqTrans.w);
	jacobi_RotV_dq(1, 2) = 2.0f * (  dqTrans.x);
	jacobi_RotV_dq(1, 3) = 2.0f * (  dqTrans.y);

	//int i = 2.0f * (-dqTrans.w * dqRot.z - dqTrans.x * dqRot.y + dqTrans.y * dqRot.x + dqTrans.z * dqRot.w);
	jacobi_RotV_dq(2, 0) = 2.0f * (   dqTrans.y);
	jacobi_RotV_dq(2, 1) = 2.0f * (-  dqTrans.x);
	jacobi_RotV_dq(2, 2) = 2.0f * (-  dqTrans.w);
	jacobi_RotV_dq(2, 3) = 2.0f * (   dqTrans.z);

	return jacobi_RotV_dq;
}

//convert dq to translation vector
inline __device__ float3x4 dq2TransVectorJacobiDQTrans(const float4& dqRot, const float4& dqTrans)
{
	float3x4 jacobi_RotV_dq;

	//int i = 2.0f * (-dqTrans.w * dqRot.x + dqTrans.x * dqRot.w - dqTrans.y * dqRot.z + dqTrans.z * dqRot.y);
	jacobi_RotV_dq(0, 0) = 2.0f * (   dqRot.w);
	jacobi_RotV_dq(0, 1) = 2.0f * (-  dqRot.z);
	jacobi_RotV_dq(0, 2) = 2.0f * (   dqRot.y);
	jacobi_RotV_dq(0, 3) = 2.0f * (-  dqRot.x);


	//int i = 2.0f * (-dqTrans.w * dqRot.y + dqTrans.x * dqRot.z + dqTrans.y * dqRot.w - dqTrans.z * dqRot.x);
	jacobi_RotV_dq(1, 0) = 2.0f * (   dqRot.z);
	jacobi_RotV_dq(1, 1) = 2.0f * (   dqRot.w);
	jacobi_RotV_dq(1, 2) = 2.0f * (-  dqRot.x);
	jacobi_RotV_dq(1, 3) = 2.0f * (-  dqRot.y);

	//int i = 2.0f * (-dqTrans.w * dqRot.z - dqTrans.x * dqRot.y + dqTrans.y * dqRot.x + dqTrans.z * dqRot.w);
	jacobi_RotV_dq(2, 0) = 2.0f * (-   dqRot.y);
	jacobi_RotV_dq(2, 1) = 2.0f * (    dqRot.x);
	jacobi_RotV_dq(2, 2) = 2.0f * (    dqRot.w);
	jacobi_RotV_dq(2, 3) = 2.0f * (-   dqRot.z);

	return jacobi_RotV_dq;
}
