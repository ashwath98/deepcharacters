
//==============================================================================================//

#include <cuda_runtime.h> 
#include "PerspectiveNPointGPUOpDataGrad.h"
#include "../../CudaUtils/CameraUtil.h"
#include "../../CudaUtils/cuda_SimpleMatrixUtil.h"

//==============================================================================================//

__global__ void computePerspectiveNPointGPUOpGradDevice(PerspectiveNPointGPUOpGradData data)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < data.numberOfBatches * data.numberOfMarkers)
	{
		int markerId = idx % data.numberOfMarkers;
		int batchId = (idx - markerId) / data.numberOfMarkers;

		float3x3 inverseMatrix;
		inverseMatrix(0, 0) = data.d_inverseMatrix[batchId * 9 + 0];
		inverseMatrix(0, 1) = data.d_inverseMatrix[batchId * 9 + 1];
		inverseMatrix(0, 2) = data.d_inverseMatrix[batchId * 9 + 2];

		inverseMatrix(1, 0) = data.d_inverseMatrix[batchId * 9 + 3];
		inverseMatrix(1, 1) = data.d_inverseMatrix[batchId * 9 + 4];
		inverseMatrix(1, 2) = data.d_inverseMatrix[batchId * 9 + 5];

		inverseMatrix(2, 0) = data.d_inverseMatrix[batchId * 9 + 6];
		inverseMatrix(2, 1) = data.d_inverseMatrix[batchId * 9 + 7];
		inverseMatrix(2, 2) = data.d_inverseMatrix[batchId * 9 + 8];

		float3x3 J;
		J.setAll(0.f);

		float3x3 identity;
		identity.setAll(0.f);
		identity(0, 0) = 1.f;
		identity(1, 1) = 1.f;
		identity(2, 2) = 1.f;

		/////////////////////
		//Construct J and matrix to be inverted 
		/////////////////////

		for (int c = 0; c < data.numberOfCameras; c++)
		{
			int offset0 = batchId * data.numberOfCameras * data.numberOfMarkers * 3 + c * data.numberOfMarkers * 3 + markerId * 3;

			float3 d_d = make_float3(data.d_d[offset0 + 0],data.d_d[offset0 + 1],data.d_d[offset0 + 2]);

			bool predConfGood = length(d_d) != 0.f;
			float predConfGoodFloat = float(predConfGood);
			
			float3x3 dcj_dcjT = float3x3::tensorProduct(d_d,d_d);
		
			J = J + ( dcj_dcjT - identity) * predConfGoodFloat;
		}

		/////////////////////
		//Output
		/////////////////////

		float3 globalTransGrad = make_float3(data.d_inputGlobalTranslationGrad[batchId * 3 + 0],
											 data.d_inputGlobalTranslationGrad[batchId * 3 + 1],
											 data.d_inputGlobalTranslationGrad[batchId * 3 + 2]);

		float3 outMarker3DDerivative = (inverseMatrix * J).getTranspose() * globalTransGrad;

		int offset1 = batchId * data.numberOfMarkers * 3 + markerId * 3;

		float backpropGradientFloat = float(data.backpropGradient);
		
		data.d_outputMarker3DGrad[offset1 + 0] = backpropGradientFloat * outMarker3DDerivative.x;
		data.d_outputMarker3DGrad[offset1 + 1] = backpropGradientFloat * outMarker3DDerivative.y;
		data.d_outputMarker3DGrad[offset1 + 2] = backpropGradientFloat * outMarker3DDerivative.z;
	
	}
}

//==============================================================================================//

extern "C" void computePerspectiveNPointGPUOpGradGPU(PerspectiveNPointGPUOpGradData& data)
{
	const int numberOfBlocks = ((data.numberOfBatches* data.numberOfMarkers) + THREADS_PER_BLOCK_PNPGPUOPGRAD - 1) / THREADS_PER_BLOCK_PNPGPUOPGRAD;
	computePerspectiveNPointGPUOpGradDevice << <numberOfBlocks, THREADS_PER_BLOCK_PNPGPUOPGRAD >> >(data);
}