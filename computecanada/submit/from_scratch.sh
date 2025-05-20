#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1      # Use --gres to specify GPU type
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=11:59:00
#SBATCH --output=../output/%j.out
#SBATCH --account=def-dclausi
#SBATCH --mail-user=jnoat92@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

# def-l44xu-ab
# salloc --time=2:59:0 --account=def-dclausi --nodes 1 --tasks-per-node=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32G

set -e

module purge
module load  StdEnv/2020 python/3.10.2
module load gcc/9.3.0 opencv/4.8.0 cuda/11.7
echo "loading module done"
source ~/env_mmselfsup/bin/activate
echo "Activating virtual environment done"

export WANDB_MODE=offline
export WANDB_DATA_DIR='/home/jnoat92/scratch/wandb'
export WANDB_SERVICE_WAIT=60


cd /home/jnoat92/projects/rrg-dclausi/ai4arctic/sea-ice-mmseg
echo "Finetune Config file: $1"

srun --kill-on-bad-exit=1 --cpus-per-task=6 python tools/train.py $1
# srun --ntasks=2 --gres=gpu:2  --kill-on-bad-exit=1 --cpus-per-task=6 python tools/train.py $1 \
#                               --launcher slurm

# Extract the base name without extension
base_name_mmseg=$(basename "$1" .py)
# CHECKPOINT_mmseg=$(cat work_dirs/$base_name_mmseg/last_checkpoint)
CHECKPOINT_mmseg=$(find work_dirs/$base_name_mmseg/ -type f -name '*best_combined_score*' | head -n 1)
echo "mmseg checkpoint $CHECKPOINT_mmseg"

srun --kill-on-bad-exit=1 --cpus-per-task=6 python tools/test.py $1 $CHECKPOINT_mmseg \
                                --out work_dirs/$base_name_mmseg/ --show-dir work_dirs/$base_name_mmseg/    \
# srun --ntasks=2 --gres=gpu:2  --kill-on-bad-exit=1 --cpus-per-task=6 python tools/test.py $1 $CHECKPOINT_mmseg \
#                                 --out work_dirs/$base_name_mmseg/ --show-dir work_dirs/$base_name_mmseg/    \
#                                 --launcher slurm
