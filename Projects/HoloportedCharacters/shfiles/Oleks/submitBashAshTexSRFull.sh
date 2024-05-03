#!/bin/bash

#SBATCH -p gpu22
#SBATCH -t 0-12:00:00
#SBATCH -o output/Christian/MeshSR4KTexFull_%a.out
#SBATCH -e output/Christian/MeshSR4KTexFull_%a.out
#SBATCH -a 1-2%1
#SBATCH --gres gpu:1

# setup the slurm
. ./slurmSetup.sh

#start training
echo "Run the python"
echo $PWD

python Bash.py --slurmId $SLURM_ARRAY_JOB_ID --config Configs/Oleks/oleks_sr_tex_4k.sh
