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
#include "cuda_SimpleMatrixUtil.h"

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

__inline__ __device__ void getRayCuda(float2& p, float3& ro, float3& rd, int cameraId, float4* invCamExtrinsics, float4* invCamProj)
{
	float4 o = make_float4(invCamExtrinsics[4 * cameraId + 0].w, invCamExtrinsics[4 * cameraId + 1].w, invCamExtrinsics[4 * cameraId + 2].w, invCamExtrinsics[4 * cameraId + 3].w);
	o /= o.w;
	ro = make_float3(o.x, o.y, o.z);

	rd = normalize(backprojectPixelCuda(make_float3(p.x, p.y, 1000.f), cameraId, invCamProj) - ro);
}

//==============================================================================================//

__inline__ __device__ float3 rayRayIntersection(float3 l1, float3 o1, float3 l2, float3 o2 )
{
	float h = length(cross(l1, o2 - o1));
	float k = length(cross(-l1, l2));

	if (fabs(k) < 0.00000001f)
		return make_float3(0.f,0.f,0.f);

	float3 res = o2 + (h / k) * -l2;
	return res;
}

//==============================================================================================//

/*
Computes the ray triangle intersection and returns the barycentric coordinates
*/
inline __device__  bool rayTriangleIntersect(float3 orig, float3 dir, float3 v0, float3 v1, float3 v2, float &t, float &a, float &b)
{
	// compute plane's normal
	float3  v0v1 = v1 - v0;
	float3  v0v2 = v2 - v0;

	// no need to normalize
	float3  N = cross(v0v1, v0v2); // N 

	/////////////////////////////
	// Step 1: finding P
	/////////////////////////////

	// check if ray and plane are parallel ?
	float NdotRayDirection = dot(dir, N);
	if (fabs(NdotRayDirection) < 0.0000001f) // almost 0 
	{
		return false; // they are parallel so they don't intersect ! 
	}

	// compute t (equation 3)
	t = (dot(v0, N) - dot(orig, N)) / NdotRayDirection;
	
	// compute the intersection point using equation 1
	float3 P = orig + t * dir;

	/////////////////////////////
	// Step 2: inside-outside test
	/////////////////////////////

	float3 C; // vector perpendicular to triangle's plane 

	// edge 0
	float3 edge0 = v1 - v0;
	float3 vp0 = P - v0;
	C = cross(edge0, vp0);
	if (dot(N, C) < 0)
	{
		a = b = 1000000.0;
		return false;
	}
	// edge 1
	float3 edge1 = v2 - v1;
	float3 vp1 = P - v1;
	C = cross(edge1, vp1);
	if ((a = dot(N, C)) < 0)
	{
		a = b = 1000000.0;
		return false;
	}
	// edge 2
	float3 edge2 = v0 - v2;
	float3 vp2 = P - v2;
	C = cross(edge2, vp2);

	if ((b = dot(N, C)) < 0)
	{
		a = b = 1000000.0;
		return false;
	}

	float denom = dot(N, N);
	a /= denom;
	b /= denom;

	if (isnan(a) || isnan(b) || isinf(a) || isinf(b))
	{
		a = b = 1000000.0;
		return false;
	}

	return true; // this ray hits the triangle 
}

//==============================================================================================//

__inline__ __device__ float p2eDistance(float3 p, float3 lP1, float3 lP2)
{
	float3 edgeNorm = normalize(lP2 - lP1);
	float edgeLength = length(lP2 - lP1);

	float t = dot(p - lP1, edgeNorm);

	if (t < 0.f)
		return length(p - lP1);
	else if (t > edgeLength)
		return length(p - lP2);
	else
		return length(cross(lP1 - p, edgeNorm));
}

//==============================================================================================//

__inline__ __device__ float2 pointOnLineBarycentrics(float3 p, float3 lP1, float3 lP2)
{
	float3 edgeNorm = normalize(lP2 - lP1);
	float edgeLength = length(lP2 - lP1);

	float t = dot(p - lP1, edgeNorm);

	if (t < 0.f)
		return make_float2(1.f, 0.f);
	else if (t > edgeLength)
		return make_float2(0.f, 1.f);
	else
	{
		float3 ap = p - lP1;
		float3 ab = lP2 - lP1;
		float3 pointOnLine = lP1 + (dot(ap, ab) / dot(ab, ab)) * ab;

		float el = length(ab);
		float l1 = length(pointOnLine - lP1);

		if (el == 0.f) //degenerated triangle
			return make_float2(0.f, 0.f);
		else
			return make_float2(1.f - l1 / el, l1 / el);
	}
}

//==============================================================================================//