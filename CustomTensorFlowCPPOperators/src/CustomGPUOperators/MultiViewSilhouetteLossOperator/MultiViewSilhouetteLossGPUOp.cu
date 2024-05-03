
//==============================================================================================//

#include <cuda_runtime.h> 
#include "MultiViewSilhouetteLossGPUOpData.h"
#include "../../CudaUtils/cudaUtil.h"
#include "../../CudaUtils/IndexHelper.h"

//==============================================================================================//

__global__ void computeMultiViewSilhouetteLossGPUOpDevice(MultiViewSilhouetteLossGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (data.numberOfBatches* data.numberOfCameras*data.numberOfPoints))
	{
		//(batches || cameras || vertices )
		int3 index3D = index1DTo3D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints,idx);
		int batchId = index3D.x;
		int cameraId = index3D.y;
		int pointId = index3D.z;

		//get multi view crop

		float2 borderOffset = make_float2(	data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 0)], 
											data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 1)]);
		float scaleFloat =					data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 2)];
		float2 croppOffsetMin = make_float2(data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 3)], 
											data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 4)]);
		float2 croppOffsetMax = make_float2(data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 5)], 
											data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 6)]);

		//get 2D point and normal
		float pointU = data.d_inputPointsImageSpace[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, 2, batchId, cameraId, pointId, 0)];
		float pointV = data.d_inputPointsImageSpace[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, 2, batchId, cameraId, pointId, 1)];

		//in view 
		bool inView = pointU-2 > croppOffsetMin.x && pointU + 2 < croppOffsetMax.x && pointV - 2 > croppOffsetMin.y && pointV + 2 < croppOffsetMax.y;

		if (!inView)
		{
			//(batches || cameras || vertices )
			data.d_outputDTImageGradients[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, 2, batchId, cameraId, pointId, 0)] = 0.f;
			data.d_outputDTImageGradients[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, 2, batchId, cameraId, pointId, 1)] = 0.f;
			data.d_outputMVSilResidual[index3DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, batchId, cameraId, pointId)] = 0.f;
			return;
		}

		//bring the point to cropped rescaled image space
		float2 pointFloat = make_float2(pointU, pointV);
		pointFloat = ((pointFloat - croppOffsetMin) / scaleFloat) + borderOffset;
		int2 point = make_int2(pointFloat.x, pointFloat.y);

		float normalU = data.d_inputNormalsImageSpace[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, 2, batchId, cameraId, pointId, 0)];
		float normalV = data.d_inputNormalsImageSpace[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, 2, batchId, cameraId, pointId, 1)];
		float2 normal = make_float2(normalU, normalV);

		//get the DT value
		//(batches || cameras || columns || rows )
		int offset = batchId * data.numberOfCameras * data.frameResolutionU * data.frameResolutionV + cameraId * data.frameResolutionU * data.frameResolutionV;
		
		float dtValue = (int) data.d_inputDTImage[offset + point.y * data.frameResolutionU + point.x];
		if (dtValue >= 127) dtValue = (dtValue - 127) * 2;

		float f00 = (int)data.d_inputDTImage[offset + (point.y - 1) * data.frameResolutionU + (point.x - 1)];
		float f01 = (int)data.d_inputDTImage[offset + (point.y - 1) * data.frameResolutionU + (point.x + 0)];
		float f02 = (int)data.d_inputDTImage[offset + (point.y - 1) * data.frameResolutionU + (point.x + 1)];

		float f10 = (int)data.d_inputDTImage[offset + (point.y + 0) * data.frameResolutionU + (point.x - 1)];
		float f12 = (int)data.d_inputDTImage[offset + (point.y + 0) * data.frameResolutionU + (point.x + 1)];

		float f20 = (int)data.d_inputDTImage[offset + (point.y + 1) * data.frameResolutionU + (point.x - 1)];
		float f21 = (int)data.d_inputDTImage[offset + (point.y + 1) * data.frameResolutionU + (point.x + 0)];
		float f22 = (int)data.d_inputDTImage[offset + (point.y + 1) * data.frameResolutionU + (point.x + 1)];

		if (f00 >= 127) f00 = (f00 - 127) * 2;
		if (f01 >= 127) f01 = (f01 - 127) * 2;
		if (f02 >= 127) f02 = (f02 - 127) * 2;

		if (f10 >= 127) f10 = (f10 - 127) * 2;
		if (f12 >= 127) f12 = (f12 - 127) * 2;

		if (f20 >= 127) f20 = (f20 - 127) * 2;
		if (f21 >= 127) f21 = (f21 - 127) * 2;
		if (f22 >= 127) f22 = (f22 - 127) * 2;

		float dC_du = (1.f / 8.f)*
			((-1.f) * f00 + (+1.f) * f02 +
			(-2.f) * f10 + (+2.f) * f12 +
				(-1.f) * f20 + (+1.f) * f22
				);

		float dC_dv = (1.f / 8.f)*
			((-1.f) * f00 + (-2.f) * f01 + (-1.f) * f02 +
			(+1.f) * f20 + (+2.f) * f21 + (+1.f) * f22
				);


		float2 DTImageNormal = make_float2(dC_du, dC_dv);
		float2 DTImageNormalUnSwaped = make_float2(dC_du, dC_dv);

		bool insideForeground =((int) data.d_inputDTImage[offset + point.y * data.frameResolutionU + point.x]) <= 127;

		//check vanishing dt gradients
		bool nonVanishingGradients = (DTImageNormal.x != 0 || DTImageNormal.y != 0);
		float nonVanishingGradientsFloat = (int)nonVanishingGradients;

		//get boundary 
		bool isBoundary = data.d_inputIsBoundary[batchId*data.numberOfCameras*data.numberOfPoints + cameraId*data.numberOfPoints + pointId];
		float isBoundaryFloat = (int)isBoundary;

		if (insideForeground)
		{
			DTImageNormal = -DTImageNormal;
		}

		float dotNormals = dot(DTImageNormal, normal);

		bool normalsAgree = dotNormals >= 0.f;

		if (!normalsAgree && insideForeground)
		{
			DTImageNormalUnSwaped = -DTImageNormalUnSwaped;
		}

		dtValue *= isBoundaryFloat;
		dtValue *= nonVanishingGradientsFloat;

		DTImageNormalUnSwaped *= isBoundaryFloat;
		DTImageNormalUnSwaped *= nonVanishingGradientsFloat;

		//layer output 
		data.d_outputDTImageGradients[batchId*data.numberOfCameras*data.numberOfPoints * 2 + cameraId*data.numberOfPoints * 2 + pointId * 2 + 0] = DTImageNormalUnSwaped.x * (1.f / (255.f));
		data.d_outputDTImageGradients[batchId*data.numberOfCameras*data.numberOfPoints * 2 + cameraId*data.numberOfPoints * 2 + pointId * 2 + 1] = DTImageNormalUnSwaped.y * (1.f / (255.f));

		data.d_outputMVSilResidual[batchId * data.numberOfCameras * data.numberOfPoints + cameraId * data.numberOfPoints + pointId] = dtValue	* (1.f / (255.f));
	}
}

//==============================================================================================//

__global__ void computeMultiViewSilhouetteLoss2GPUOpDevice(MultiViewSilhouetteLossGPUOpData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (data.numberOfBatches* data.numberOfCameras*data.frameResolutionU*data.frameResolutionV))
	{
		int4 index4D = index1DTo4D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, idx);
		int batchId			= index4D.x;
		int cameraId		= index4D.y;
		int VId				= index4D.z;
		int UId				= index4D.w;

		//get multi view crop
		float2 croppedImagePos = make_float2(UId, VId);

		//check if it is boundary pixel
		float dtValue = (int) data.d_inputDTImage[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, batchId, cameraId, VId, UId)];
		int neighbourCounter = 0;
	
		int closestVertexId = -1;

		if (dtValue == 127.f)
		{
			//check if really boundary pixel
			for (int j = -2; j <= 2; j++)
			{
				for (int i = -2; i <= 2; i++)
				{ 
					int neighbourIndex = index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, batchId, cameraId, VId + j, UId + i);
					if(neighbourIndex != -1)
					{
						if (((int)data.d_inputDTImage[neighbourIndex] )== 127.f)
						{
							neighbourCounter++;
						}
					}
				}
			}

			//is really boundary
			if (neighbourCounter >= 4)
			{
				//get image pos in global image
				
				float2 borderOffset = make_float2  (data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 0)], 
												    data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 1)]);
				float scaleFloat =                  data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 2)];
				float2 croppOffsetMin = make_float2(data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 3)], 
												    data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 4)]);
				float2 imagePos = ((croppedImagePos - borderOffset) * scaleFloat) + croppOffsetMin;
				float closestVertex = 30.f * scaleFloat;
		
				//search for closest boundary point
				for (int v = 0; v < data.numberOfPoints; v++)
				{
					bool isBoundary = data.d_inputIsBoundary[index3DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, batchId, cameraId, v)];

					if (isBoundary)
					{
						float pointU = data.d_inputPointsImageSpace[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, 2, batchId, cameraId, v, 0)];
						float pointV = data.d_inputPointsImageSpace[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, 2, batchId, cameraId, v, 1)];
						float2 vertexPos = make_float2(pointU, pointV);

						float distance = length(vertexPos - imagePos);
						if (distance < closestVertex)
						{
							closestVertexId = v;
							closestVertex = distance;
						}
					}
				}
			
				//end search 
				if (closestVertexId >= 0)
				{
					//get closest point infos
					float closestVertexPosU = data.d_inputPointsImageSpace[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, 2, batchId, cameraId, closestVertexId, 0)];
					float closestVertexPosV = data.d_inputPointsImageSpace[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, 2, batchId, cameraId, closestVertexId, 1)];
					float2 closestVertexPos = make_float2(closestVertexPosU, closestVertexPosV);

					float2 loss = (closestVertexPos - imagePos) / (30.f * scaleFloat);

					//layer output 
					data.d_outputMVSilResidual1[index5DTo1D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, 2, batchId, cameraId, VId, UId, 0)] = loss.x;
					data.d_outputMVSilResidual1[index5DTo1D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, 2, batchId, cameraId, VId, UId, 1)] = loss.y;
					data.d_outputClosestVertexIds[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, batchId, cameraId, VId, UId)] = closestVertexId;
				}
			}
			
		}
		if(neighbourCounter < 4 || closestVertexId < 0)
		{
			//layer output 
			data.d_outputMVSilResidual1[index5DTo1D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, 2, batchId, cameraId, VId, UId, 0)] = 0.f;
			data.d_outputMVSilResidual1[index5DTo1D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, 2, batchId, cameraId, VId, UId, 1)] = 0.f;
			data.d_outputClosestVertexIds[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, batchId, cameraId, VId, UId)] = -1;
		}
	}
}

//==============================================================================================//

extern "C" void computeMultiViewSilhouetteLossGPUOpGPU(MultiViewSilhouetteLossGPUOpData& data)
{
	const int numberOfBlocks = ((data.numberOfBatches* data.numberOfCameras*data.numberOfPoints) + THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP - 1) / THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP;
	computeMultiViewSilhouetteLossGPUOpDevice << <numberOfBlocks, THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP >> >(data);

	const int numberOfBlocks1 = ((data.numberOfBatches* data.numberOfCameras*data.frameResolutionU*data.frameResolutionV) + THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP - 1) / THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP;
	computeMultiViewSilhouetteLoss2GPUOpDevice << <numberOfBlocks1, THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP >> >(data);
}

//=================================================================s=============================//