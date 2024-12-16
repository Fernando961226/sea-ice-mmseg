#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=4 # request a GPU
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=12 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=200G
#SBATCH --time=15:59:00
#SBATCH --output=../output/%j.out
#SBATCH --account=rrg-dclausi
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

cd /home/jnoat92/projects/rrg-dclausi/ai4arctic/sea-ice-mmselfsup
echo "Pretrain Config file: $1"
# Extract the base name without extension
base_name=$(basename "$1" .py)
CHECKPOINT=$(cat work_dirs/selfsup/$base_name/last_checkpoint)
echo "mmselfsup Checkpoint $CHECKPOINT"

# ============== DOWNSTREAM TASK

cd /home/jnoat92/projects/rrg-dclausi/ai4arctic/sea-ice-mmseg
echo "Finetune Config file: $2"

# # CHECKPOINT=(/home/jnoat92/projects/rrg-dclausi/ai4arctic/sea-ice-mmseg/work_dirs/mae_ai4arctic_ds2_pt_80_ft_20/iter_40000.pth)
# srun --ntasks=4 --gres=gpu:4  --kill-on-bad-exit=1 --cpus-per-task=12 python tools/train.py $2 \
#                                 --cfg-options   model.backbone.init_cfg.checkpoint=${CHECKPOINT} \
#                                                 model.backbone.init_cfg.type='Pretrained' \
#                                                 model.backbone.init_cfg.prefix='backbone.' \
#                                 --launcher slurm 
# srun --ntasks=4 --gres=gpu:4  --kill-on-bad-exit=1 --cpus-per-task=12 python tools/train.py $2 --resume --launcher slurm
srun --ntasks=4 --gres=gpu:4  --kill-on-bad-exit=1 --cpus-per-task=12 python tools/train.py $2 --launcher slurm

# Extract the base name without extension
base_name_mmseg=$(basename "$2" .py)
# CHECKPOINT_mmseg=$(cat work_dirs/$base_name_mmseg/last_checkpoint)
CHECKPOINT_mmseg=$(find work_dirs/$base_name_mmseg/ -type f -name '*best_combined_score*' | head -n 1)
echo "mmseg checkpoint $CHECKPOINT_mmseg"

srun --ntasks=4 --gres=gpu:4  --kill-on-bad-exit=1 --cpus-per-task=12 python tools/test.py $2 $CHECKPOINT_mmseg \
                                --out work_dirs/$base_name_mmseg/ --show-dir work_dirs/$base_name_mmseg/    \
                                --launcher slurm