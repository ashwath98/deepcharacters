#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 2-00:00:00 
#SBATCH -o output/out-%j.out
#SBATCH -e output/err-%j.err
#SBATCH --gres gpu:1

# call your program here
echo "using GPU ${CUDA_VISIBLE_DEVICES}"
cd ../build/Linux
make -j 32