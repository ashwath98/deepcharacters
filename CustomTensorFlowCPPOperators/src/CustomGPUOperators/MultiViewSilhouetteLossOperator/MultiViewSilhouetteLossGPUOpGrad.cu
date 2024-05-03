
//==============================================================================================//

#include <cuda_runtime.h> 
#include "MultiViewSilhouetteLossGPUOpGradData.h"
#include "../../CudaUtils/cudaUtil.h"
#include "../../CudaUtils/IndexHelper.h"

//==============================================================================================//

__global__ void computeMultiViewSilhouetteLossGPUOpGradDevice(MultiViewSilhouetteLossGPUOpGradData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (data.numberOfBatches*data.numberOfCameras*data.numberOfPoints))
	{
		int pointId = (idx % (data.numberOfCameras*data.numberOfPoints)) % data.numberOfPoints;
		int cameraId = ((idx - pointId) % (data.numberOfCameras*data.numberOfPoints)) / data.numberOfPoints;
		int batchId = (idx - cameraId * data.numberOfPoints - pointId) / (data.numberOfCameras * data.numberOfPoints);

		float dtImageGradientU = data.d_inputDTImageGrad[batchId * data.numberOfCameras * data.numberOfPoints * 2 + cameraId * data.numberOfPoints * 2 + pointId * 2 + 0];
		float dtImageGradientV = data.d_inputDTImageGrad[batchId * data.numberOfCameras * data.numberOfPoints * 2 + cameraId * data.numberOfPoints * 2 + pointId * 2 + 1];

		float dtLossGrad = data.d_inputDTLossGrad[batchId * data.numberOfCameras * data.numberOfPoints + cameraId * data.numberOfPoints + pointId];
		
		data.d_outputPointsImageSpaceGrad[batchId * data.numberOfCameras * data.numberOfPoints * 2 + cameraId * data.numberOfPoints * 2 + pointId * 2 + 0] = dtImageGradientU * dtLossGrad;
		data.d_outputPointsImageSpaceGrad[batchId * data.numberOfCameras * data.numberOfPoints * 2 + cameraId * data.numberOfPoints * 2 + pointId * 2 + 1] = dtImageGradientV * dtLossGrad;
	}
}

//==============================================================================================//

__global__ void computeMultiViewSilhouetteLossGPUOpGradDevice1(MultiViewSilhouetteLossGPUOpGradData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (data.numberOfBatches*data.numberOfCameras*data.frameResolutionU*data.frameResolutionV))
	{
		int4 index4D = index1DTo4D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, idx);
		int batchId		= index4D.x;
		int cameraId	= index4D.y;
		int VId			= index4D.z;
		int UId			= index4D.w;

		int closestVertex = data.d_closestVertexId[index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, batchId, cameraId, VId, UId)];

		if (closestVertex >= 0)
		{
			float scaleFloat = data.d_inputMultiViewCrops[index3DTo1D(data.numberOfBatches, data.numberOfCameras, 7, batchId, cameraId, 2)];

			float dtLossGrad1U = data.d_inputDTLossGrad1[index5DTo1D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, 2, batchId, cameraId, VId, UId, 0)];
			float dtLossGrad1V = data.d_inputDTLossGrad1[index5DTo1D(data.numberOfBatches, data.numberOfCameras, data.frameResolutionV, data.frameResolutionU, 2, batchId, cameraId, VId, UId, 1)];
			int outputOffset0 = index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, 2, batchId, cameraId, closestVertex, 0);
			int outputOffset1 = index4DTo1D(data.numberOfBatches, data.numberOfCameras, data.numberOfPoints, 2, batchId, cameraId, closestVertex, 1);
			atomicAdd(data.d_outputPointsImageSpaceGrad + outputOffset0, dtLossGrad1U / (30.f * scaleFloat));
			atomicAdd(data.d_outputPointsImageSpaceGrad + outputOffset1, dtLossGrad1V / (30.f * scaleFloat));
		}
	}
}

//==============================================================================================//

extern "C" void computeMultiViewSilhouetteLossGPUOpGradGPU(MultiViewSilhouetteLossGPUOpGradData& data)
{
	//model to data
	const int numberOfBlocks = ((data.numberOfBatches*data.numberOfCameras*data.numberOfPoints) + THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP - 1) / THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP;
	computeMultiViewSilhouetteLossGPUOpGradDevice << <numberOfBlocks, THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP>> >(data);

	//data to model
	const int numberOfBlocks1 = ((data.numberOfBatches*data.numberOfCameras*data.frameResolutionU*data.frameResolutionV) + THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP - 1) / THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP;
	computeMultiViewSilhouetteLossGPUOpGradDevice1 << <numberOfBlocks1, THREADS_PER_BLOCK_MultiViewSilhouetteLossGPUOP >> >(data);
}

//==============================================================================================//