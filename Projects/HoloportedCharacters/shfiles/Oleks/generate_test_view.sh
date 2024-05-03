#!/bin/bash

#SBATCH -p gpu22
#SBATCH -t 0-2:00:00
#SBATCH -o output/ChristianSpiral_%A.out
#SBATCH -e output/ChristianSpiral_%A.out
#SBATCH -a 1
#SBATCH --gres gpu:1


#start training
echo "Run the python"
echo $PWD

python testing_scripts/test_data_loader_spiral_more_start_end.py results/Oleks_Spiral_Full_4K  Configs/Oleks/oleks_tex_test_4K.sh /CT/ashwath2/static00/DatasetRelease/Calibrations/Oleks/rotation.calibration output/oleks_rotation_infinite_test.mp4 0 200

