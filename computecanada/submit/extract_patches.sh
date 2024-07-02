#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task=64 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=50000M
#SBATCH --time=0:30:00
#SBATCH --output=../output/%j.out
#SBATCH --account=def-dclausi
#SBATCH --mail-user=j3hsiao@uwaterloo.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

# salloc --time=3:00:0 --account=rrg-dclausi --nodes 1 --tasks-per-node=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32G
set -e

module purge
module load  StdEnv/2020 python/3.10.2
module load gcc/9.3.0 opencv/4.8.0 cuda/11.7
echo "loading module done"

source ~/env_mmsegmentation/bin/activate

echo "Activating virtual environment done"

cd $HOME/projects/rrg-dclausi/$USER/sea-ice-mmseg/tools/AI4Artic_dataset

# # python patch_dataset_creator_raw.py --downsampling 1
# python patch_dataset_creator_raw.py --downsampling 2
# python patch_dataset_creator_raw.py --downsampling 3
# python patch_dataset_creator_raw.py --downsampling 4
# python patch_dataset_creator_raw.py --downsampling 5
# python patch_dataset_creator_raw.py --downsampling 6
# python patch_dataset_creator_raw.py --downsampling 7
# python patch_dataset_creator_raw.py --downsampling 8
# python patch_dataset_creator_raw.py --downsampling 9
python patch_dataset_creator_raw.py --downsampling 10


