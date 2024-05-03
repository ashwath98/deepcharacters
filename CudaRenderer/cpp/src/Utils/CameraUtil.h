//==============================================================================================//
// Classname:
//      CameraUtil
//
//==============================================================================================//
// Description:
//      Basic camera operations for the GPU like projection and applying extrinsics
//
//==============================================================================================//

#pragma once 

//==============================================================================================//

#include <cutil_inline.h>
#include <cutil_math.h>

//==============================================================================================//

__inline__ __device__ int2 projectPoint(float3 point, float3 row1, float3 row2, float3 row3)
{
	//perspective projection
	float3 dot1 = point * row1;
	float3 dot2 = point * row2;
	float3 dot3 = point * row3;

	float x = dot1.x + dot1.y + dot1.z;
	float y = dot2.x + dot2.y + dot2.z;
	float z = dot3.x + dot3.y + dot3.z;

	//perspective divide
	if (z > 0.0000001f)
	{
		x /= z;
		y /= z;
		return make_int2((x + 0.5f), (y + 0.5f));
	}
	else
	{
		z = 0.00001f;
		x /= z;
		y /= z;
		return make_int2((x + 0.5f), (y + 0.5f));
	}
}

//==============================================================================================//

__inline__ __device__ float2 projectPointFloat(float3 point, float3 row1, float3 row2, float3 row3)
{
	//perspective projection
	float3 dot1 = point * row1;
	float3 dot2 = point * row2;
	float3 dot3 = point * row3;

	float x = dot1.x + dot1.y + dot1.z;
	float y = dot2.x + dot2.y + dot2.z;
	float z = dot3.x + dot3.y + dot3.z;

	//perspective divide
	if (z > 0.0000001f)
	{
		x /= z;
		y /= z;
		return make_float2(x, y);
	}
	else
	{
		z = 0.00001f;
		x /= z;
		y /= z;
		return make_float2(x, y);
	}
	
}

//==============================================================================================//

__inline__ __device__ float3 projectPointFloat3(float3 point, float3 row1, float3 row2, float3 row3)
{
	//perspective projection
	float3 dot1 = point * row1;
	float3 dot2 = point * row2;
	float3 dot3 = point * row3;

	float x = dot1.x + dot1.y + dot1.z;
	float y = dot2.x + dot2.y + dot2.z;
	float z = dot3.x + dot3.y + dot3.z;

	//perspective divide
	if (z > 0.0000001f)
	{
		x /= z;
		y /= z;
		return make_float3(x, y, z);
	}
	else 
	{
		z = 0.00001f;
		x /= z;
		y /= z;
		return make_float3(x, y, z);
	}
}

//==============================================================================================//

__inline__ __device__ float2 projectPointFloat(float3* intrinsicMatrix, float3 point)
{
	float3 row1 = intrinsicMatrix[0];
	float3 row2 = intrinsicMatrix[1];
	float3 row3 = intrinsicMatrix[2];

	//perspective projection
	float3 dot1 = point * row1;
	float3 dot2 = point * row2;
	float3 dot3 = point * row3;

	float x = dot1.x + dot1.y + dot1.z;
	float y = dot2.x + dot2.y + dot2.z;
	float z = dot3.x + dot3.y + dot3.z;

	//perspective divide
	if (z > 0.0000001f)
	{
		x /= z;
		y /= z;
		return make_float2(x, y);
	}
	else
	{
		z = 0.00001f;
		x /= z;
		y /= z;
		return make_float2(x, y);
	}
}

//==============================================================================================//

__inline__ __device__ float3 projectPointFloat3(float3* intrinsicMatrix, float3 point)
{
	float3 row1 = intrinsicMatrix[0];
	float3 row2 = intrinsicMatrix[1];
	float3 row3 = intrinsicMatrix[2];

	//perspective projection
	float3 dot1 = point * row1;
	float3 dot2 = point * row2;
	float3 dot3 = point * row3;

	float x = dot1.x + dot1.y + dot1.z;
	float y = dot2.x + dot2.y + dot2.z;
	float z = dot3.x + dot3.y + dot3.z;

	//perspective divide
	if (z > 0.0000001f)
	{
		x /= z;
		y /= z;
		return make_float3(x, y, z);
	}
	else
	{
		z = 0.00001f;
		x /= z;
		y /= z;
		return make_float3(x, y, z);
	}
}

//==============================================================================================//

__inline__ __device__ float3 getCamSpacePoint(float4* extrinsicMatrix, float3 point)
{
	//get vertex in cam space
	float4 homogCamSpaceVertex = make_float4(point.x, point.y, point.z, 1.f);

	float3 camSpaceVertex = make_float3(
		dot(extrinsicMatrix[0], homogCamSpaceVertex),
		dot(extrinsicMatrix[1], homogCamSpaceVertex),
		dot(extrinsicMatrix[2], homogCamSpaceVertex)
		);

	return camSpaceVertex;
}

//==============================================================================================//

__inline__ __device__ float3 getCamSpaceVector(float4* extrinsicMatrix, float3 vector)
{
	//get vector in cam space
	float4 homogCamSpaceVector = make_float4(vector.x, vector.y, vector.z, 0.f);

	float3 camSpaceVector = make_float3(
		dot(extrinsicMatrix[0], homogCamSpaceVector),
		dot(extrinsicMatrix[1], homogCamSpaceVector),
		dot(extrinsicMatrix[2], homogCamSpaceVector)
		);
	
	return camSpaceVector;
}

//==============================================================================================//

__inline__ __device__ float3 backprojectPixelCuda(float3 p, int cameraId, float4* invCamProj)
{
	const float3 tp = make_float3(p.x * p.z, p.y * p.z, p.z);
	float4 tpHomo = make_float4(tp.x, tp.y, tp.z, 1.f);

	float4 temp = make_float4(
		dot(invCamProj[4 * cameraId + 0], tpHomo),
		dot(invCamProj[4 * cameraId + 1], tpHomo),
		dot(invCamProj[4 * cameraId + 2], tpHomo),
		dot(invCamProj[4 * cameraId + 3], tpHomo)
	);

	return make_float3(temp.x, temp.y, temp.z);
}

//==============================================================================================//

__inline__ __device__ float3 backprojectPixelCuda(float3 p, float4* invCamProj)
{
	const float3 tp = make_float3(p.x * p.z, p.y * p.z, p.z);
	float4 tpHomo = make_float4(tp.x, tp.y, tp.z, 1.f);

	float4 temp = make_float4(
		dot(invCamProj[0], tpHomo),
		dot(invCamProj[1], tpHomo),
		dot(invCamProj[2], tpHomo),
		dot(invCamProj[3], tpHomo)
	);

	return make_float3(temp.x, temp.y, temp.z);
}

//==============================================================================================//

__inline__ __device__ void getRayCuda(float2& p, float3& ro, float3& rd, int cameraId, float4* invCamExtrinsics, float4* invCamProj)
{
	float4 o = make_float4(invCamExtrinsics[4 * cameraId + 0].w, invCamExtrinsics[4 * cameraId + 1].w, invCamExtrinsics[4 * cameraId + 2].w, invCamExtrinsics[4 * cameraId + 3].w);
	o /= o.w;
	ro = make_float3(o.x, o.y, o.z);

	rd = normalize(backprojectPixelCuda(make_float3(p.x, p.y, 1000.f), cameraId, invCamProj) - ro);
}

//==============================================================================================//

__inline__ __device__ void getRayCuda2(float2& p, float3& ro, float3& rd, float4* invCamExtrinsics, float4* invCamProj)
{
	float4 o = make_float4(invCamExtrinsics[0].w, invCamExtrinsics[1].w, invCamExtrinsics[2].w, invCamExtrinsics[3].w);
	o /= o.w;
	ro = make_float3(o.x, o.y, o.z);

	rd = normalize(backprojectPixelCuda(make_float3(p.x, p.y, 1000.f), invCamProj) - ro);
}