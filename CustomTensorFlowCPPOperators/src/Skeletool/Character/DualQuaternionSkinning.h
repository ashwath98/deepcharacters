//==============================================================================================//
// Classname:
//      DualQuaternionSkinning
//
//==============================================================================================//
// Description:
//      DualQuaternion is basically used to blend rotation matrices for skinning based
//		character deformation (for more information read dual quaternion skinning paper)
//		This class is basically the GPU version to make it faster.
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <cuda_runtime.h>

#include "skinnedcharacter.h"
#include "DualQuaternion.h"
#include "DualQuaternionSkinningInput.h"

#include <cutil.h>
#include <cutil_inline_runtime.h>

//==============================================================================================//

class DualQuaternionSkinning
{

	//functions

	public:

		DualQuaternionSkinning(skinnedcharacter* character);
		virtual ~DualQuaternionSkinning();

		inline virtual float4* getH_dualQuaternions(){ return h_dualQuaternions; }


		virtual void dualQuaternionSkinning();
		virtual void dualQuaternionPlusDisplacementSkinning();
		virtual void displacementSkinning();
		virtual void inverseDisplacementSkinning();
		virtual void initializeDQSkinning();
		virtual void initializePreviousDisplacement(float3* d_previousDisplacement);

	private:
		
		virtual void setupGPUQuaternion();
		virtual void setupInverseGPUQuaternion();

	//variables

	public:

	private:

		skinnedcharacter* character;
		int numJoints;
		DualQuaternionSkinningInput dualQuaternionSkinningInput;
		float4* h_dualQuaternions;

		int	numSkinningConnections;

		int*			h_numSkinningNodes;
		int*			h_indexSkinningNodes;
		int*			h_skinningNodes;
		float*			h_skinningNodeWeights;

		bool previousDisplacementInitialized;
};

//==============================================================================================//
