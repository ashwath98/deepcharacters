#!/bin/bash

#SBATCH -p cpu20
#SBATCH -c 40
#SBATCH -t 0-12:00:00
#SBATCH -o logs/Oleks/frames_%a.out
#SBATCH -e logs/Oleks/frames_%a.out
#SBATCH -a 0-116%116


#start training
echo "Run the python"
echo $PWD

# get the time limit for the job
SBATCH_TIMELIMIT=$(squeue -j $SLURM_ARRAY_JOB_ID -o %l)0
arrIN=(${SBATCH_TIMELIMIT// / })

#${arrIN[1]}

python create_frames.py $SLURM_ARRAY_TASK_ID Oleks loose training Oleks 100 200
echo "Done"
