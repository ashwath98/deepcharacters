//==============================================================================================//
// Classname:
//      RendererUtil 
//
//==============================================================================================//
// Description:
//      Basic operations for rendering and gradients for rendering 
//
//==============================================================================================//

#pragma once 

//==============================================================================================//

#include <cutil_inline.h>
#include <cutil_math.h>
#include "CameraUtil.h"
#include "IndexHelper.h"
#include "cuda_SimpleMatrixUtil.h"

//==============================================================================================//

/*
Computes the ray triangle intersection and returns the barycentric coordinates
*/
inline __device__  bool rayTriangleIntersect(float3 orig, float3 dir, float3 v0, float3 v1, float3 v2, float &t, float &a, float &b)
{
	//just to make it numerically more stable
	v0 = v0 / 1000.f;
	v1 = v1 / 1000.f;
	v2 = v2 / 1000.f;
	orig = orig / 1000.f;

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
	// compute d parameter using equation 2
	float d = dot(N, v0);

	// compute t (equation 3)
	t = (dot(v0, N) - dot(orig, N)) / NdotRayDirection;
	// check if the triangle is in behind the ray
	if (t < 0)
	{
		return false; // the triangle is behind 
	}
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
		return false;
	}
	// edge 1
	float3 edge1 = v2 - v1;
	float3 vp1 = P - v1;
	C = cross(edge1, vp1);
	if ((a = dot(N, C)) < 0)
	{
		return false;
	}
	// edge 2
	float3 edge2 = v0 - v2;
	float3 vp2 = P - v2;
	C = cross(edge2, vp2);

	if ((b = dot(N, C)) < 0)
	{
		return false;
	}

	float denom = dot(N, N);
	a /= denom;
	b /= denom;

	return true; // this ray hits the triangle 
}

//==============================================================================================//

/*
Computes the per pixel barycentric coordinates
*/
inline __device__ float3 uv2barycentric(float u, float v, float3 v0, float3 v1, float3 v2, float4* invExtrinsics, float4* invProjection)
{
	float3 o = make_float3(0.f, 0.f, 0.f);
	float3 d = make_float3(0.f, 0.f, 0.f);

	float2 pixelPos = make_float2(u, v);

	getRayCuda2(pixelPos, o, d, invExtrinsics, invProjection);

	float t, a, b, c;

	bool intersect;
	intersect = rayTriangleIntersect(o, d, v0, v1, v2, t, a, b);

	if (!intersect)
		a = b = c = -1.f;
	else
		c = 1.f - a - b;

	return make_float3(a, b, t);
}

//==============================================================================================//

/*
Takes albedo color, normal direction and shading coefficients and computes the shaded color
*/
inline __device__ float3 getShading(float3 color, float3 dir, const float *shCoeffs)
{
	float3 dirSq = dir * dir;
	float3 shadedColor;

	shadedColor.x  = shCoeffs[0];
	shadedColor.x += shCoeffs[1] * dir.y;
	shadedColor.x += shCoeffs[2] * dir.z;
	shadedColor.x += shCoeffs[3] * dir.x;
	shadedColor.x += shCoeffs[4] * (dir.x * dir.y);
	shadedColor.x += shCoeffs[5] * (dir.z * dir.y);
	shadedColor.x += shCoeffs[6] * (3.f * dirSq.z - 1.f);
	shadedColor.x += shCoeffs[7] * (dir.x * dir.z);
	shadedColor.x += shCoeffs[8] * (dirSq.x - dirSq.y);
	shadedColor.x = shadedColor.x * color.x;

	shadedColor.y  = shCoeffs[9 + 0];
	shadedColor.y += shCoeffs[9 + 1] * dir.y;
	shadedColor.y += shCoeffs[9 + 2] * dir.z;
	shadedColor.y += shCoeffs[9 + 3] * dir.x;
	shadedColor.y += shCoeffs[9 + 4] * (dir.x * dir.y);
	shadedColor.y += shCoeffs[9 + 5] * (dir.z * dir.y);
	shadedColor.y += shCoeffs[9 + 6] * (3.f * dirSq.z - 1.f);
	shadedColor.y += shCoeffs[9 + 7] * (dir.x * dir.z);
	shadedColor.y += shCoeffs[9 + 8] * (dirSq.x - dirSq.y);
	shadedColor.y = shadedColor.y * color.y;

	shadedColor.z  = shCoeffs[18 + 0];
	shadedColor.z += shCoeffs[18 + 1] * dir.y;
	shadedColor.z += shCoeffs[18 + 2] * dir.z;
	shadedColor.z += shCoeffs[18 + 3] * dir.x;
	shadedColor.z += shCoeffs[18 + 4] * (dir.x * dir.y);
	shadedColor.z += shCoeffs[18 + 5] * (dir.z * dir.y);
	shadedColor.z += shCoeffs[18 + 6] * (3.f * dirSq.z - 1.f);
	shadedColor.z += shCoeffs[18 + 7] * (dir.x * dir.z);
	shadedColor.z += shCoeffs[18 + 8] * (dirSq.x - dirSq.y);
	shadedColor.z = shadedColor.z * color.z;
	return shadedColor;
}
//==============================================================================================//

/*
Computes the illumination from the surface normal and the lighting coefficients
*/
__inline__ __device__ float3 getIllum(float3 dir, const float *shCoeffs)
{
	float3 dirSq = dir * dir;
	float3 light;

	light.x  = shCoeffs[0];
	light.x += shCoeffs[1] * dir.y;
	light.x += shCoeffs[2] * dir.z;
	light.x += shCoeffs[3] * dir.x;
	light.x += shCoeffs[4] * (dir.x * dir.y);
	light.x += shCoeffs[5] * (dir.z * dir.y);
	light.x += shCoeffs[6] * (3.f * dirSq.z - 1.f);
	light.x += shCoeffs[7] * (dir.x * dir.z);
	light.x += shCoeffs[8] * (dirSq.x - dirSq.y);

	light.y  = shCoeffs[9 + 0];
	light.y += shCoeffs[9 + 1] * dir.y;
	light.y += shCoeffs[9 + 2] * dir.z;
	light.y += shCoeffs[9 + 3] * dir.x;
	light.y += shCoeffs[9 + 4] * (dir.x * dir.y);
	light.y += shCoeffs[9 + 5] * (dir.z * dir.y);
	light.y += shCoeffs[9 + 6] * (3.f * dirSq.z - 1.f);
	light.y += shCoeffs[9 + 7] * (dir.x * dir.z);
	light.y += shCoeffs[9 + 8] * (dirSq.x - dirSq.y);

	light.z  = shCoeffs[18 + 0];
	light.z += shCoeffs[18 + 1] * dir.y;
	light.z += shCoeffs[18 + 2] * dir.z;
	light.z += shCoeffs[18 + 3] * dir.x;
	light.z += shCoeffs[18 + 4] * (dir.x * dir.y);
	light.z += shCoeffs[18 + 5] * (dir.z * dir.y);
	light.z += shCoeffs[18 + 6] * (3.f * dirSq.z - 1.f);
	light.z += shCoeffs[18 + 7] * (dir.x * dir.z);
	light.z += shCoeffs[18 + 8] * (dirSq.x - dirSq.y);
	return light;
}

//==============================================================================================//

/*
Extracts the rotation matrix from the full extrinsics matrix
*/
__device__ inline mat3x3 getRotationMatrix(float4* d_T)
{
	mat3x3 TE;
	TE(0, 0) = d_T[0].x;
	TE(0, 1) = d_T[0].y;
	TE(0, 2) = d_T[0].z;
	TE(1, 0) = d_T[1].x;
	TE(1, 1) = d_T[1].y;
	TE(1, 2) = d_T[1].z;
	TE(2, 0) = d_T[2].x;
	TE(2, 1) = d_T[2].y;
	TE(2, 2) = d_T[2].z;
	return TE;
}

//==============================================================================================//

/*
d_shadedColor / d_albedo
*/
__inline__ __device__ void getJCoAl(mat3x3 &JCoAl, float3 pixLight)
{
	JCoAl.setZero();
	JCoAl(0, 0) = pixLight.x;
	JCoAl(1, 1) = pixLight.y;
	JCoAl(2, 2) = pixLight.z;
}

//==============================================================================================//

/*
d_albedo / d_vertex_colors
*/
__inline__ __device__ void getJAlVc(mat3x9 &JAlVc, float3 bcc)
{
	JAlVc.setZero();
	JAlVc(0, 0) = bcc.x;
	JAlVc(1, 1) = bcc.x;
	JAlVc(2, 2) = bcc.x;

	JAlVc(0, 3) = bcc.y;
	JAlVc(1, 4) = bcc.y;
	JAlVc(2, 5) = bcc.y;

	JAlVc(0, 6) = bcc.z;
	JAlVc(1, 7) = bcc.z;
	JAlVc(2, 8) = bcc.z;
}

//==============================================================================================//

/*
d_projection / d_position
*/
__inline__ __device__ void getJProjection(mat2x3 &JProjection, float3 globalPosition, float3* intrinsics, float4* extrinsics)
{
	JProjection.setZero();

	mat4x3 dP;
	dP.setZero();
	dP(0, 0) = 1.f;
	dP(1, 1) = 1.f;
	dP(2, 2) = 1.f;

	mat3x4 E;
	E(0, 0) = extrinsics[0].x;
	E(0, 1) = extrinsics[0].y;
	E(0, 2) = extrinsics[0].z;
	E(0, 3) = extrinsics[0].w;

	E(1, 0) = extrinsics[1].x;
	E(1, 1) = extrinsics[1].y;
	E(1, 2) = extrinsics[1].z;
	E(1, 3) = extrinsics[1].w;

	E(2, 0) = extrinsics[2].x;
	E(2, 1) = extrinsics[2].y;
	E(2, 2) = extrinsics[2].z;
	E(2, 3) = extrinsics[2].w;

	mat3x3 I;
	I(0, 0) = intrinsics[0].x;
	I(0, 1) = intrinsics[0].y;
	I(0, 2) = intrinsics[0].z;

	I(1, 0) = intrinsics[1].x;
	I(1, 1) = intrinsics[1].y;
	I(1, 2) = intrinsics[1].z;

	I(2, 0) = intrinsics[2].x;
	I(2, 1) = intrinsics[2].y;
	I(2, 2) = intrinsics[2].z;

	mat4x1 PP;
	PP(0, 0) = globalPosition.x;
	PP(1, 0) = globalPosition.y;
	PP(2, 0) = globalPosition.z;
	PP(3, 0) = 1.f;
	mat3x1 P = I * E * PP;

	mat2x3 dDivide;
	dDivide(0, 0) = 1.f / P(2, 0);
	dDivide(0, 1) = 0.f;
	dDivide(0, 2) = -P(0, 0) / (P(2, 0) * P(2, 0));

	dDivide(1, 0) = 0.f;
	dDivide(1, 1) = 1.f / P(2, 0);
	dDivide(1, 2) = -P(1, 0) / (P(2, 0) * P(2, 0));

	if(fabs(P(2, 0)) > 0.0001f)
		JProjection = dDivide * I * E * dP;
}
//==============================================================================================//

/*
d_shadedColor / d_lighting
*/
__inline__ __device__ void getJCoLi(mat3x3 &JCoLi, float3 pixAlb)
{
	JCoLi.setZero();
	JCoLi(0, 0) = pixAlb.x;
	JCoLi(1, 1) = pixAlb.y;
	JCoLi(2, 2) = pixAlb.z;
}

//==============================================================================================//

/*
d_lighting / d_lightingCoeffs
*/
__inline__ __device__ void getJLiGm(mat3x9 &JLiGm, int rgb, float3 pixNorm)
{
	JLiGm.setZero();

	JLiGm(rgb, 0) = 1;
	JLiGm(rgb, 1) = pixNorm.y;
	JLiGm(rgb, 2) = pixNorm.z;
	JLiGm(rgb, 3) = pixNorm.x;
	JLiGm(rgb, 4) = pixNorm.x * pixNorm.y;
	JLiGm(rgb, 5) = pixNorm.z * pixNorm.y;
	JLiGm(rgb, 6) = 3 * pixNorm.z*pixNorm.z - 1;
	JLiGm(rgb, 7) = pixNorm.x * pixNorm.z;
	JLiGm(rgb, 8) = ((pixNorm.x * pixNorm.x) - (pixNorm.y*pixNorm.y));
}

//==============================================================================================//

/*
d_lighting / d_normalizedNormal
*/
__inline__ __device__ void getJLiNo(mat3x3 &JLiNo, float3 dir, const float* shCoeff)
{
	JLiNo.setZero();
	for (int i = 0; i < 3; i++) 
	{
		JLiNo(i, 0) =    shCoeff[(i * 9) + 3] +
						(shCoeff[(i * 9) + 4] * dir.y) +
						(shCoeff[(i * 9) + 7] * dir.z) +
						(shCoeff[(i * 9) + 8] * 2 * dir.x);

		JLiNo(i, 1) =    shCoeff[(i * 9) + 1] +
						(shCoeff[(i * 9) + 4] * dir.x) +
						(shCoeff[(i * 9) + 5] * dir.z) +
						(shCoeff[(i * 9) + 8] * -2.f * dir.y);

		JLiNo(i, 2) =    shCoeff[(i * 9) + 2] +
						(shCoeff[(i * 9) + 5] * dir.y) +
						(shCoeff[(i * 9) + 6] * 6 * dir.z) +
						(shCoeff[(i * 9) + 7] * dir.x);
	}
}

//==============================================================================================//

/*
d_normalizedNormal / d_unnormalizedNormal
*/
__inline__ __device__ void getJNoNu(mat3x3 &JNoNu, float3 un_vec, float norm)
{
	float norm_p2 = norm * norm;
	float norm_p3 = norm_p2 * norm;

	JNoNu(0, 0) = (norm_p2 - (un_vec.x*un_vec.x)) / (norm_p3);
	JNoNu(1, 1) = (norm_p2 - (un_vec.y*un_vec.y)) / (norm_p3);
	JNoNu(2, 2) = (norm_p2 - (un_vec.z*un_vec.z)) / (norm_p3);

	JNoNu(0, 1) = -(un_vec.x*un_vec.y) / norm_p3;
	JNoNu(1, 0) = JNoNu(0, 1);

	JNoNu(0, 2) = -(un_vec.x*un_vec.z) / norm_p3;
	JNoNu(2, 0) = JNoNu(0, 2);

	JNoNu(1, 2) = -(un_vec.y*un_vec.z) / norm_p3;
	JNoNu(2, 1) = JNoNu(1, 2);
}

//==============================================================================================//

/*
d_unnormalizedNormal / d_v_k
*/
__inline__ __device__ void getJ_vk(mat3x3 &J, mat3x1 vj, mat3x1 vi)
{
	float3 temp3;

	mat3x1 Ix(make_float3(1.f, 0.f, 0.f));
	mat3x1 Iy(make_float3(0.f, 1.f, 0.f));
	mat3x1 Iz(make_float3(0.f, 0.f, 1.f));
	
	//J2
	mat3x3 J2;
	mat3x1 diff2 = vj - vi;

	temp3 = cross(diff2, Ix);
	J2(0, 0) = temp3.x;
	J2(1, 0) = temp3.y;
	J2(2, 0) = temp3.z;

	temp3 = cross(diff2, Iy);
	J2(0, 1) = temp3.x;
	J2(1, 1) = temp3.y;
	J2(2, 1) = temp3.z;

	temp3 = cross(diff2, Iz);
	J2(0, 2) = temp3.x;
	J2(1, 2) = temp3.y;
	J2(2, 2) = temp3.z;

	J = J2 ;
}

//==============================================================================================//

/*
d_unnormalizedNormal / d_v_j
*/
__inline__ __device__ void getJ_vj(mat3x3 &J, mat3x1 vk, mat3x1 vi)
{
	float3 temp3;

	mat3x1 Ix(make_float3(1.f, 0.f, 0.f));
	mat3x1 Iy(make_float3(0.f, 1.f, 0.f));
	mat3x1 Iz(make_float3(0.f, 0.f, 1.f));

	//J1
	mat3x3 J1;
	mat3x1 diff1 = vk - vi;

	temp3 = cross(Ix, diff1);
	J1(0, 0) = temp3.x;
	J1(1, 0) = temp3.y;
	J1(2, 0) = temp3.z;

	temp3 = cross(Iy, diff1);
	J1(0, 1) = temp3.x;
	J1(1, 1) = temp3.y;
	J1(2, 1) = temp3.z;

	temp3 = cross(Iz, diff1);
	J1(0, 2) = temp3.x;
	J1(1, 2) = temp3.y;
	J1(2, 2) = temp3.z;

	J = J1  ;
}

//==============================================================================================//

/*
d_unnormalizedNormal / d_v_i
*/
__inline__ __device__ void getJ_vi(mat3x3 &J, mat3x1 vk, mat3x1 vj, mat3x1 vi)
{
	float3 temp3;

	mat3x1 IxNeg(make_float3(-1.f, 0.f, 0.f));
	mat3x1 IyNeg(make_float3(0.f, -1.f, 0.f));
	mat3x1 IzNeg(make_float3(0.f, 0.f, -1.f));

	//J1
	mat3x3 J1;
	mat3x1 diff1 = vj - vi;

	temp3 = cross(diff1, IxNeg);
	J1(0, 0) = temp3.x;		
	J1(1, 0) = temp3.y;
	J1(2, 0) = temp3.z;

	temp3 = cross(diff1, IyNeg);
	J1(0, 1) = temp3.x;
	J1(1, 1) = temp3.y;
	J1(2, 1) = temp3.z;

	temp3 = cross(diff1, IzNeg);
	J1(0, 2) = temp3.x;
	J1(1, 2) = temp3.y;
	J1(2, 2) = temp3.z;

	//J2
	mat3x3 J2;
	mat3x1 diff2 = vk - vi;

	temp3 = cross(IxNeg, diff2);
	J2(0, 0) = temp3.x;
	J2(1, 0) = temp3.y;
	J2(2, 0) = temp3.z;

	temp3 = cross(IyNeg, diff2);
	J2(0, 1) = temp3.x;
	J2(1, 1) = temp3.y;
	J2(2, 1) = temp3.z;

	temp3 = cross(IzNeg, diff2);
	J2(0, 2) = temp3.x;
	J2(1, 2) = temp3.y;
	J2(2, 2) = temp3.z;

	J = (J1 + J2 );
}

//==============================================================================================//

/*
d_albedo / d_barycentricCoords
*/
__inline__ __device__ void getJAlBc(mat3x3 &JAlBc, float3 vertexCol0, float3 vertexCol1, float3 vertexCol2)
{
	JAlBc.setZero();

	JAlBc(0, 0) = vertexCol0.x;
	JAlBc(0, 1) = vertexCol1.x;
	JAlBc(0, 2) = vertexCol2.x;
	JAlBc(1, 0) = vertexCol0.y;
	JAlBc(1, 1) = vertexCol1.y;
	JAlBc(1, 2) = vertexCol2.y;
	JAlBc(2, 0) = vertexCol0.z;
	JAlBc(2, 1) = vertexCol1.z;
	JAlBc(2, 2) = vertexCol2.z;
}

//==============================================================================================//

/*
Image gradient helper
*/
__inline__ __device__ mat3x2 imageGradient(const float3* image, float2 point, int imageWidth, int imageHeight, int filterSize)
{
	mat3x2 dIdUV;
	dIdUV.setZero();

	float3 dI_du = make_float3(0.f, 0.f, 0.f);
	float3 dI_dv = make_float3(0.f, 0.f, 0.f);

	if (   point.x >= (filterSize+1 )
		&& point.y >= (filterSize + 1 )
		&& point.x <  ( imageWidth - (filterSize + 1) )
		&& point.y <  (imageHeight - (filterSize + 1)))
	{
		float normalizationFactor = 0.f;

		for (int y = -filterSize; y <= filterSize; y++)
		{
			for (int x = -filterSize; x <= filterSize; x++)
			{
				float3 I = image[(int)((point.y + y) * imageWidth + (point.x + x))];

				float denom = ((float)(x*x + y*y));
				float xFloat = x;
				float yFloat = y;

				float G_u = 0.f;
				float G_v = 0.f;

				if (denom != 0.f)
				{
					G_u = xFloat / denom; //sobel filter for U
					G_v = yFloat / denom; //sobel filter for V
				}
				
				dI_du += I * G_u;
				dI_dv += I * G_v;

				normalizationFactor += fabs(G_u);		
			}
		}

		dI_du /= normalizationFactor;
		dI_dv /= normalizationFactor;
		
		dIdUV(0, 0) = dI_du.x;
		dIdUV(1, 0) = dI_du.y;
		dIdUV(2, 0) = dI_du.z;

		dIdUV(0, 1) = dI_dv.x;
		dIdUV(1, 1) = dI_dv.y;
		dIdUV(2, 1) = dI_dv.z;
	}

	return dIdUV;
}

//==============================================================================================//

/*
d_albedo / d_barycentricCoords
*/
__inline__ __device__ void getJAlTexBc(mat3x3 &JAlBc, const float* d_textureMap, float2 uv, float2 tc0, float2 tc1, float2 tc2, int imageWidth, int imageHeight, int filterSize)
{
	JAlBc.setZero();

	mat3x2 dIdUV = imageGradient((const float3*)d_textureMap, uv, imageWidth, imageHeight, filterSize);

	mat2x3 dUVdabc;
	dUVdabc(0, 0) = tc0.x * imageWidth;
	dUVdabc(1, 0) = tc0.y * imageHeight;

	dUVdabc(0, 1) = tc1.x * imageWidth;
	dUVdabc(1, 1) = tc1.y * imageHeight;

	dUVdabc(0, 2) = tc2.x * imageWidth;
	dUVdabc(1, 2) = tc2.y * imageHeight;

	JAlBc = dIdUV * dUVdabc;
}
//==============================================================================================//

/*
d_unnormalizedNormal / d_barycentricCoords
*/
__inline__ __device__ void getJNoBc(mat3x3 &JNoBc, float3 N0, float3 N1, float3 N2)
{
	JNoBc(0, 0) = N0.x;
	JNoBc(0, 1) = N1.x;
	JNoBc(0, 2) = N2.x;

	JNoBc(1, 0) = N0.y;
	JNoBc(1, 1) = N1.y;
	JNoBc(1, 2) = N2.y;

	JNoBc(2, 0) = N0.z;
	JNoBc(2, 1) = N1.z;
	JNoBc(2, 2) = N2.z;
}

//==============================================================================================//

/*
d_barycentricCoords / d_vertexPositons
*/
inline __device__  void dJBCDVerpos(mat3x9& dJBC, float3 orig, float3 dir, float3 v0, float3 v1, float3 v2)
{
	dJBC.setZero();

	float3  v0v1 = v1 - v0;
	float3  v0v2 = v2 - v0;

	float3  N = cross(v0v1, v0v2); 
	float denom = dot(N, N);
	float NdotRayDirection = dot(dir, N);

	//to avoid division by very small number (essentially it checks if the ray and the face normal are not close to perpendicular to each other) == angle more than 87 degree
	if (fabs(dot(normalize(dir), normalize(N))) < 0.001f || fabs(denom * denom) < 0.001f)
	{
		return;
	}

	float d = dot(N, v0);

	float t = (dot(v0, N) - dot(orig, N)) / NdotRayDirection;

	float3 P = orig + t * dir;

	float3 edge1 = v2 - v1;
	float3 vp1 = P - v1;
	float3 C1= cross(edge1, vp1);

	float3 edge2 = v0 - v2;
	float3 vp2 = P - v2;
	float3 C2 = cross(edge2, vp2);

	mat3x3 identity;
	identity.setIdentity();

	//
	float3 dN[9];

	mat3x3 dN_v0;
	mat3x3 dN_v1;
	mat3x3 dN_v2;

	mat3x1 v0Mat;
	v0Mat(0, 0) = v0.x;
	v0Mat(1, 0) = v0.y;
	v0Mat(2, 0) = v0.z;

	mat3x1 v1Mat;
	v1Mat(0, 0) = v1.x;
	v1Mat(1, 0) = v1.y;
	v1Mat(2, 0) = v1.z;

	mat3x1 v2Mat;
	v2Mat(0, 0) = v2.x;
	v2Mat(1, 0) = v2.y;
	v2Mat(2, 0) = v2.z;

	getJ_vi(dN_v0, v2Mat, v1Mat, v0Mat);
	getJ_vj(dN_v1, v2Mat, v0Mat);
	getJ_vk(dN_v2, v1Mat, v0Mat);

	dN[0].x = dN_v0(0, 0);
	dN[0].y = dN_v0(1, 0);
	dN[0].z = dN_v0(2, 0);

	dN[1].x = dN_v0(0, 1);
	dN[1].y = dN_v0(1, 1);
	dN[1].z = dN_v0(2, 1);

	dN[2].x = dN_v0(0, 2);
	dN[2].y = dN_v0(1, 2);
	dN[2].z = dN_v0(2, 2);
	
	dN[3].x = dN_v1(0, 0);
	dN[3].y = dN_v1(1, 0);
	dN[3].z = dN_v1(2, 0);

	dN[4].x = dN_v1(0, 1);
	dN[4].y = dN_v1(1, 1);
	dN[4].z = dN_v1(2, 1);

	dN[5].x = dN_v1(0, 2);
	dN[5].y = dN_v1(1, 2);
	dN[5].z = dN_v1(2, 2);

	dN[6].x = dN_v2(0, 0);
	dN[6].y = dN_v2(1, 0);
	dN[6].z = dN_v2(2, 0);

	dN[7].x = dN_v2(0, 1);
	dN[7].y = dN_v2(1, 1);
	dN[7].z = dN_v2(2, 1);

	dN[8].x = dN_v2(0, 2);
	dN[8].y = dN_v2(1, 2);
	dN[8].z = dN_v2(2, 2);

	//
	float3 C[2];
	C[0] = C1;
	C[1] = C2;

	//
	float3 dE[2][9];
	dE[0][0] = make_float3(0.f, 0.f, 0.f);
	dE[0][1] = make_float3(0.f, 0.f, 0.f);
	dE[0][2] = make_float3(0.f, 0.f, 0.f);
	dE[0][3] = make_float3(-1.f, 0.f, 0.f);
	dE[0][4] = make_float3(0.f, -1.f, 0.f);
	dE[0][5] = make_float3(0.f, 0.f, -1.f);
	dE[0][6] = make_float3(1.f, 0.f, 0.f);
	dE[0][7] = make_float3(0.f, 1.f, 0.f);
	dE[0][8] = make_float3(0.f, 0.f, 1.f);

	dE[1][0] = make_float3(1.f, 0.f, 0.f);
	dE[1][1] = make_float3(0.f, 1.f, 0.f);
	dE[1][2] = make_float3(0.f, 0.f, 1.f);
	dE[1][3] = make_float3(0.f, 0.f, 0.f);
	dE[1][4] = make_float3(0.f, 0.f, 0.f);
	dE[1][5] = make_float3(0.f, 0.f, 0.f);
	dE[1][6] = make_float3(-1.f, 0.f, 0.f);
	dE[1][7] = make_float3(0.f, -1.f, 0.f);
	dE[1][8] = make_float3(0.f, 0.f, -1.f);

	//
	float3 vp[2];
	vp[0] = vp1;
	vp[1] = vp2;
	
	//
	float3 E[2]; 
	E[0] = edge1; //v2 - v1
	E[1] = edge2; //v0 - v2
	
	//
	float3 dV[3][9];

	dV[0][0] = make_float3(1.f, 0.f, 0.f);
	dV[0][1] = make_float3(0.f, 1.f, 0.f);
	dV[0][2] = make_float3(0.f, 0.f, 1.f);
	dV[0][3] = make_float3(0.f, 0.f, 0.f);
	dV[0][4] = make_float3(0.f, 0.f, 0.f);
	dV[0][5] = make_float3(0.f, 0.f, 0.f);
	dV[0][6] = make_float3(0.f, 0.f, 0.f);
	dV[0][7] = make_float3(0.f, 0.f, 0.f);
	dV[0][8] = make_float3(0.f, 0.f, 0.f);

	dV[1][0] = make_float3(0.f, 0.f, 0.f);
	dV[1][1] = make_float3(0.f, 0.f, 0.f);
	dV[1][2] = make_float3(0.f, 0.f, 0.f);
	dV[1][3] = make_float3(1.f, 0.f, 0.f);
	dV[1][4] = make_float3(0.f, 1.f, 0.f);
	dV[1][5] = make_float3(0.f, 0.f, 1.f);
	dV[1][6] = make_float3(0.f, 0.f, 0.f);
	dV[1][7] = make_float3(0.f, 0.f, 0.f);
	dV[1][8] = make_float3(0.f, 0.f, 0.f);

	dV[2][0] = make_float3(0.f, 0.f, 0.f);
	dV[2][1] = make_float3(0.f, 0.f, 0.f);
	dV[2][2] = make_float3(0.f, 0.f, 0.f);
	dV[2][3] = make_float3(0.f, 0.f, 0.f);
	dV[2][4] = make_float3(0.f, 0.f, 0.f);
	dV[2][5] = make_float3(0.f, 0.f, 0.f);
	dV[2][6] = make_float3(1.f, 0.f, 0.f);
	dV[2][7] = make_float3(0.f, 1.f, 0.f);
	dV[2][8] = make_float3(0.f, 0.f, 1.f);
	
	//
	float dt[9];
	for (int var = 0; var < 9; var++)
	{
		dt[var] = (-1.f /(dot(N,dir)*dot(N, dir))) * dot(dN[var],dir) * dot((v0-orig),N) +  (1.f /(dot(N,dir))) * (dot(dV[0][var],N)+ dot((v0-orig),dN[var]));
	}

	//
	for (int abc = 0; abc < 2; abc++)
	{
		for (int var = 0; var < 9; var++)
		{
			float tmp = dot(N, C[abc]);
			
			float dtmp = (dot(dN[var], C[abc]) + dot(N, (cross(dE[abc][var], vp[abc]) + cross(E[abc], (dt[var] * dir - dV[abc + 1][var])))));

			dJBC(abc, var) = dtmp / denom + tmp *(-1.f/(denom*denom))* (2.f * dot(N,dN[var]));
		}
	}

	//
	for (int var = 0; var < 9; var++)
	{
		dJBC(2, var) = -dJBC(0, var) - dJBC(1, var);
	}
}

//==============================================================================================//

/*
Gradient adding helper
*/
__inline__ __device__ void addGradients(mat1x3 grad, float3* d_grad)
{
	float* d_gradFloat0 = (float*)d_grad;
	float* d_gradFloat1 = (float*)d_grad + 1;
	float* d_gradFloat2 = (float*)d_grad + 2;
	atomicAdd(d_gradFloat0, grad(0, 0));
	atomicAdd(d_gradFloat1, grad(0, 1));
	atomicAdd(d_gradFloat2, grad(0, 2));
}

//==============================================================================================//

/*
Gradient adding helper
*/
__inline__ __device__ void addGradients9(mat1x9 grad, float* d_grad)
{
	for (int ii = 0; ii < 9; ii++)
		atomicAdd(&d_grad[ii], grad(0, ii));
}

//==============================================================================================//

/*
Gradient adding helper
*/
__inline__ __device__ void addGradients9I(mat9x1 grad, float3* d_grad, int3 index)
{
	//if (index.x == 1)
	{
		atomicAdd(&d_grad[index.x].x, grad(0, 0));
		atomicAdd(&d_grad[index.x].y, grad(1, 0));
		atomicAdd(&d_grad[index.x].z, grad(2, 0));
	}
	//if (index.y ==1)
	{
		atomicAdd(&d_grad[index.y].x, grad(3, 0));
		atomicAdd(&d_grad[index.y].y, grad(4, 0));
		atomicAdd(&d_grad[index.y].z, grad(5, 0));
	}
	//if (index.z == 1)
	{
		atomicAdd(&d_grad[index.z].x, grad(6, 0));
		atomicAdd(&d_grad[index.z].y, grad(7, 0));
		atomicAdd(&d_grad[index.z].z, grad(8, 0));
	}
}
