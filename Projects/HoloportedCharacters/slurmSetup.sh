#!/bin/bash

# echo cuda
echo "using GPU ${CUDA_VISIBLE_DEVICES}"

# check mode
if [ "$SLURM_ARRAY_JOB_ID" == "" ]
then
      echo "Interactive mode used!"
      SLURM_ARRAY_JOB_ID="3"
else
      echo "Sbatch mode used!"
fi
echo "Slurm array job id: ${SLURM_ARRAY_JOB_ID}"

# go into code backup folder
echo "Go into code backup folder"
cd SlurmCodeBackUp

# check if copy of code for slurm run already exists otherwise create dir
echo "Check if copy of code for slurm run already exists otherwise create dir"
if [ -d "$SLURM_ARRAY_JOB_ID" ]
then
      if [ $SLURM_ARRAY_JOB_ID == 3 ]
      then
            echo "Directory exists but interactive mode is used so copy again!"
            SLURM_CONTINUED=0
      else
            echo "Directory exists and sbatch mode is used"
            SLURM_CONTINUED=1
      fi
else
      echo "Directory DOES NOT exist"
      SLURM_CONTINUED=0
      mkdir "$SLURM_ARRAY_JOB_ID"
fi

# go down the folder hiearchy
echo "Go into code copy dir: ${SLURM_ARRAY_JOB_ID} and further down the hierarchy"
cd "$SLURM_ARRAY_JOB_ID"

mkdir CudaRenderer
cd CudaRenderer
mkdir cpp
cd ..
cp -r ../../../../CudaRenderer/cpp/binaries CudaRenderer/cpp/.

mkdir CustomTensorFlowCPPOperators
cp -r ../../../../CustomTensorFlowCPPOperators/binaries CustomTensorFlowCPPOperators/.

mkdir Projects
cd Projects

# copy top level files if necessary
if [ $SLURM_CONTINUED == 0 ]
then
      echo "Copy top level files"
      cp -r ../../../../AdditionalUtils .
      cp -r ../../../../Architectures .
      cp -r ../../../../CudaRenderer .
      cp -r ../../../../CustomTFOperators .
      cp -r ../../../../Studio .
fi

# go deeper
echo "Go further down"
mkdir HoloportedCharacters
cd HoloportedCharacters

# copy lowest level files if necessary
if [ $SLURM_CONTINUED == 0 ]
then
      echo "Copy low level files"
      cp -r ../../../../Utils .
      cp -r ../../../../Configs .
      cp ../../../../* .
fi