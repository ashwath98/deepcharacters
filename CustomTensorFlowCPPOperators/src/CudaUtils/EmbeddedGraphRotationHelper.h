//==============================================================================================//
// Classname:
//      EmbeddedGraphRotationHelper
//
//==============================================================================================//
// Description:
//      Calculates the derivatives for the Embedded Graph deformation prior introduced in the
//		original Embedded Graph Paper of Sorkine. The prior says that Rotation and translation
//		of the neighbour should map to the same point as the own Rotation and translation.
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#ifndef _EMBEDDEDGRAPHROTATION_HELPER_
#define _EMBEDDEDGRAPHROTATION_HELPER_

//==============================================================================================//

#include "cuda_SimpleMatrixUtil.h"
#include "../CustomGPUOperators/EmbeddedGraphOperator/EmbeddedGraphGPUOpData.h"
#include "../CustomGPUOperators/EmbeddedGraphOperator/EmbeddedGraphGPUOpGradData.h"

//==============================================================================================//

// Rotation times vector
inline __device__ float3x3 embeddedGraphEvalDerivativeRotationTimesVector(const float3x3& dRAlpha, const float3x3& dRBeta, const float3x3& dRGamma, const float3& d)
{
	float3x3 R;
	float3 b = dRAlpha*d;
	R(0, 0) = b.x; 
	R(1, 0) = b.y; 
	R(2, 0) = b.z;

	b = dRBeta *d;
	R(0, 1) = b.x;
	R(1, 1) = b.y; 
	R(2, 1) = b.z;

	b = dRGamma*d;
	R(0, 2) = b.x; 
	R(1, 2) = b.y; 
	R(2, 2) = b.z;

	return R;
}

//==============================================================================================//

// Rotation Matrix
inline __device__ float3x3 embeddedGraphEvalRMat(const float3& angles)
{
	float3x3 R;

	const float cosAlpha	= cos(angles.x);
	const float cosBeta		= cos(angles.y);
	const float cosGamma	= cos(angles.z);
	const float sinAlpha	= sin(angles.x); 
	const float sinBeta		= sin(angles.y);
	const float sinGamma	= sin(angles.z);

	R(0, 0) = cosBeta*cosGamma;
	R(0, 1) = sinAlpha*sinBeta*cosGamma - cosAlpha*sinGamma;
	R(0, 2) = cosAlpha*sinBeta*cosGamma + sinAlpha*sinGamma;

	R(1, 0) = cosBeta*sinGamma;
	R(1, 1) = sinAlpha*sinBeta*sinGamma + cosAlpha*cosGamma;
	R(1, 2) = cosAlpha*sinBeta*sinGamma - sinAlpha*cosGamma;

	R(2, 0) = -sinBeta;
	R(2, 1) = sinAlpha*cosBeta;
	R(2, 2) = cosAlpha *cosBeta;

	return R;
}

//==============================================================================================//

// Rotation Matrix
inline __device__ float3 embeddedGraphEvalRAngles(const float3x3& R)
{
	float3 eulerAngle = make_float3(0.f, 0.f, 0.f);

	float sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

	bool singular = sy < 1e-6; // If

	if (!singular)
	{
		eulerAngle.x = atan2(R(2, 1), R(2, 2));
		eulerAngle.y = atan2(-R(2, 0), sy);
		eulerAngle.z = atan2(R(1, 0), R(0, 0));
	}
	else
	{
		eulerAngle.x = atan2(-R(1, 2), R(1, 1));
		eulerAngle.y = atan2(-R(2, 0), sy);
		eulerAngle.z = 0;
	}

	return eulerAngle;
}

//==============================================================================================//

// Rotation Matrix dAlpha
inline __device__ float3x3 embeddedGraphEvalRMat_dAlpha(float3 angles)
{
	float3x3 R;

	const float cosAlpha	= cos(angles.x);
	const float cosBeta		= cos(angles.y);
	const float cosGamma	= cos(angles.z);
	const float sinAlpha	= sin(angles.x);
	const float sinBeta		= sin(angles.y);
	const float sinGamma	= sin(angles.z);

	R(0, 0) = 0.f;
	R(0, 1) =  cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma;
	R(0, 2) = -sinAlpha * sinBeta * cosGamma + cosAlpha * sinGamma;

	R(1, 0) = 0.f;
	R(1, 1) =  cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma;
	R(1, 2) = -sinAlpha * sinBeta * sinGamma - cosAlpha * cosGamma;

	R(2, 0) = 0.f;
	R(2, 1) =  cosAlpha * cosBeta; 
	R(2, 2) = -sinAlpha * cosBeta;

	return R;
}

//==============================================================================================//

// Rotation Matrix dBeta
inline __device__ float3x3 embeddedGraphEvalRMat_dBeta(float3 angles)
{
	float3x3 R;

	const float cosAlpha	= cos(angles.x);
	const float cosBeta		= cos(angles.y);
	const float cosGamma	= cos(angles.z);
	const float sinAlpha	= sin(angles.x);
	const float sinBeta		= sin(angles.y);
	const float sinGamma	= sin(angles.z);

	R(0, 0) = -sinBeta *  cosGamma;
	R(0, 1) =  sinAlpha * cosBeta * cosGamma;
	R(0, 2) =  cosAlpha * cosBeta * cosGamma;

	R(1, 0) = -sinBeta  * sinGamma;
	R(1, 1) =  sinAlpha * cosBeta   * sinGamma;
	R(1, 2) =  cosAlpha * cosBeta   * sinGamma;

	R(2, 0) = -cosBeta;
	R(2, 1) = -sinAlpha * sinBeta;
	R(2, 2) = -cosAlpha * sinBeta;

	return R;
}

//==============================================================================================//

// Rotation Matrix dGamma
inline __device__ float3x3 embeddedGraphEvalRMat_dGamma(float3 angles)
{
	float3x3 R;

	const float cosAlpha	= cos(angles.x);
	const float cosBeta		= cos(angles.y);
	const float cosGamma	= cos(angles.z);
	const float sinAlpha	= sin(angles.x);
	const float sinBeta		= sin(angles.y);
	const float sinGamma	= sin(angles.z);

	R(0, 0) = -cosBeta  * sinGamma;
	R(0, 1) = -sinAlpha * sinBeta   * sinGamma - cosAlpha * cosGamma;
	R(0, 2) = -cosAlpha * sinBeta   * sinGamma + sinAlpha * cosGamma;

	R(1, 0) = cosBeta  * cosGamma;
	R(1, 1) = sinAlpha * sinBeta   * cosGamma - cosAlpha * sinGamma;
	R(1, 2) = cosAlpha * sinBeta   * cosGamma + sinAlpha * sinGamma;

	R(2, 0) = 0.f;
	R(2, 1) = 0.f;
	R(2, 2) = 0.f;

	return R;
}

//==============================================================================================//

// Rotation Matrix dIdx
inline __device__ float3x3 embeddedGraphEvalR_dIdx(float3 angles, unsigned int idx) // 0 = alpha, 1 = beta, 2 = gamma
{
	if (idx == 0) return embeddedGraphEvalRMat_dAlpha(angles);
	else if (idx == 1) return embeddedGraphEvalRMat_dBeta(angles);
	else return embeddedGraphEvalRMat_dGamma(angles);
}

//==============================================================================================//

inline __device__ void embeddedGraphEvalDerivativeRotationMatrix(const float3& angles, float3x3& dRAlpha, float3x3& dRBeta, float3x3& dRGamma)
{
	dRAlpha = embeddedGraphEvalRMat_dAlpha(angles);
	dRBeta = embeddedGraphEvalRMat_dBeta(angles);
	dRGamma = embeddedGraphEvalRMat_dGamma(angles);
}

//==============================================================================================//

inline __device__ float3 embeddedGraphDeformVertex(EmbeddedGraphGPUOpData& data, int batchIdx, int vertexIdx, float3* deformedNormal)
{
	int vertexToNodeSize = data.d_EGVertexToNodeSizes[vertexIdx];
	int vertexToNodeOffset = data.d_EGVertexToNodeOffsets[vertexIdx];

	float3 deformedVertex = make_float3(0.f, 0.f, 0.f);
	float3 localDeformedNormal = make_float3(0.f, 0.f, 0.f);

	float3 V = data.d_baseVertices[vertexIdx];
	float3 N = data.d_baseNormals[vertexIdx];

	//skinning based deformation of the the non-rigid deformed template 
	for (int s = 0; s < vertexToNodeSize; s++)
	{
		int nodeId = data.d_EGVertexToNodeIndices[vertexToNodeOffset + s];
		float nodeWeight = data.d_EGVertexToNodeWeights[vertexToNodeOffset + s];
		int offset = batchIdx * data.numberOfNodes * 3 + nodeId * 3;
		float3 G = data.d_baseVertices[data.d_EGNodeToBaseMeshVertices[nodeId]];
	
		//R_delta
		float3 nodeEulerAngleDelta = make_float3(data.d_deltaA[offset + 0], data.d_deltaA[offset + 1], data.d_deltaA[offset + 2]);
		float3x3 RDelta = embeddedGraphEvalRMat(nodeEulerAngleDelta);

		//R_skinned
		float3 nodeEulerAngleSkinned = make_float3(data.d_skinnedA[offset + 0], data.d_skinnedA[offset + 1], data.d_skinnedA[offset + 2]);
		float3x3 RSkinned = embeddedGraphEvalRMat(nodeEulerAngleSkinned);

		//R_invTrans
		float3x3 RInvTrans = (RSkinned * RDelta ).getInverse().getTranspose();

		//t_delta
		float3 TDelta = make_float3(data.d_deltaT[offset + 0], data.d_deltaT[offset + 1], data.d_deltaT[offset + 2]);

		//t_skinned
		float3 TSkinned = make_float3(data.d_skinnedT[offset + 0], data.d_skinnedT[offset + 1], data.d_skinnedT[offset + 2]);

		//apply transformation on the vertex and normal
		deformedVertex		+= nodeWeight * (RSkinned * ( RDelta* (V - G) + G + TDelta) + TSkinned) ;
		localDeformedNormal += nodeWeight * (RInvTrans*N);
	}

	//output
	deformedNormal->x = localDeformedNormal.x;
	deformedNormal->y = localDeformedNormal.y;
	deformedNormal->z = localDeformedNormal.z;

	return deformedVertex;
}

//==============================================================================================//

#endif