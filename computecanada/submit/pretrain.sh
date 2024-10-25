#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=4 # request a GPU
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=12 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=200G
#SBATCH --time=09:59:00
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

echo "starting pretrain ..."
cd /home/jnoat92/projects/rrg-dclausi/ai4arctic/sea-ice-mmselfsup

echo "Config file: $1"
# srun --ntasks=4 --gres=gpu:4 --kill-on-bad-exit=1 --cpus-per-task=12 python tools/train.py $1 --launcher slurm --resume #--cfg-options
srun --ntasks=4 --gres=gpu:4 --kill-on-bad-exit=1 --cpus-per-task=12 python tools/train.py $1 --launcher slurm

# Extract the base name without extension
base_name=$(basename "$1" .py)
CHECKPOINT=$(cat work_dirs/selfsup/$base_name/last_checkpoint)
echo "mmselfsup Checkpoint $CHECKPOINT"

# Reconstruct sample Image
srun --ntasks=1 --gres=gpu:1 --kill-on-bad-exit=1 --cpus-per-task=12 python tools/analysis_tools/visualize_reconstruction_ai4arctic.py $1 --checkpoint $CHECKPOINT --img-path "/home/jnoat92/scratch/dataset/ai4arctic/down_scale_9X/S1A_EW_GRDM_1SDH_20180814T120158_20180814T120258_023242_0286BE_36EF_icechart_cis_SGRDIHA_20180814T1201Z_pl_a/00007.pkl" --out-file "work_dirs/selfsup/$base_name/visual_reconstruction"

