#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task=48 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=510000M
#SBATCH --time=1:59:00
#SBATCH --output=../output/%j.out
#SBATCH --account=def-dclausi
#SBATCH --mail-user=jnoat92@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

# salloc --time=3:00:0 --account=def-dclausi --nodes 1 --tasks-per-node=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32G
set -e

module purge
module load  StdEnv/2020 python/3.10.2
module load gcc/9.3.0 opencv/4.8.0 cuda/11.7
echo "loading module done"

source ~/env_mmsegmentation/bin/activate

echo "Activating virtual environment done"

cd $HOME/projects/rrg-dclausi/$USER/sea-ice-mmseg/tools/AI4Artic_dataset

python patch_dataset_creator_raw.py