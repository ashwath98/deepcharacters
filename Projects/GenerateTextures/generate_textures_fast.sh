#!/bin/bash

#SBATCH -p gpu22
#SBATCH -t 0-4:00:00
#SBATCH -o outtestOT_10000_5000.out
#SBATCH -e outtestOT_10000_5000.out
#SBATCH -a 1
#SBATCH --gres gpu:1

# setup the slurm


#start training
echo "Run the python"
echo $PWD

python texture_mapping_optimized_general.py 100 200 /CT/ashwath2/static00/Oleks/partial_textures_test/