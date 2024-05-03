#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 0-12:00:00
#SBATCH -o output/VladNew/out-%A_%a.out
#SBATCH -e output/VladNew/out-%A_%a.out
#SBATCH -a 1-7%1
#SBATCH --gres gpu:4

##SBATCH -p gpu20
##SBATCH -t 0-12:00:00
##SBATCH -o output/out-%A_%a.out
##SBATCH -e output/out-%A_%a.out
##SBATCH -a 1
##SBATCH --gres gpu:1

# setup the slurm
. ./slurmSetup.sh

#start training
echo "Run the python"
echo $PWD

python Bash.py --slurmId $SLURM_ARRAY_JOB_ID --config Configs/Oleks/oleks_sr_tex_4k.sh
