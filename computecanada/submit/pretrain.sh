#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --gpus-per-node=2      # Use --gres to specify GPU type
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=09:59:0
#SBATCH --output=../output/%j.out
#SBATCH --account=rrg-dclausi
#SBATCH --mail-user=jnoat92@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

# def-l44xu-ab
# salloc --time=2:59:0 --account=def-dclausi --nodes 1 --tasks-per-node=1 --gpus-per-node=1 --cpus-per-task=8 --mem=32G
# salloc --time=0:29:0 --account=rrg-dclausi --nodes 1 --ntasks=3 --gres=gpu:a100_3g.20gb:3 --cpus-per-task=6 --mem=62G

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
# srun --kill-on-bad-exit=1 python tools/train.py $1
srun --ntasks=2 --gres=gpu:2 --cpus-per-task=6 --kill-on-bad-exit=1 python tools/train.py $1 --launcher slurm #--resume #--cfg-options

# Extract the base name without extension
base_name=$(basename "$1" .py)
CHECKPOINT=$(cat work_dirs/selfsup/$base_name/last_checkpoint)
echo "mmselfsup Checkpoint $CHECKPOINT"

# Reconstruct sample Image
# srun --kill-on-bad-exit=1 python tools/analysis_tools/visualize_reconstruction_ai4arctic.py $1 --checkpoint $CHECKPOINT --img-path "/home/jnoat92/scratch/dataset/ai4arctic/down_scale_9X/S1A_EW_GRDM_1SDH_20200424T101936_20200424T102036_032268_03BBA9_1CA8_icechart_cis_SGRDINFLD_20200424T1020Z_pl_a/00005.pkl" --out-file "work_dirs/selfsup/$base_name/visual_reconstruction"
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=6 --kill-on-bad-exit=1 python tools/analysis_tools/visualize_reconstruction_ai4arctic.py $1 --checkpoint $CHECKPOINT --img-path "/home/jnoat92/scratch/dataset/ai4arctic/down_scale_9X/S1A_EW_GRDM_1SDH_20200424T101936_20200424T102036_032268_03BBA9_1CA8_icechart_cis_SGRDINFLD_20200424T1020Z_pl_a/00005.pkl" --out-file "work_dirs/selfsup/$base_name/visual_reconstruction"

# cd /home/jnoat92/projects/rrg-dclausi/ai4arctic/sea-ice-mmselfsup
# python tools/analysis_tools/visualize_reconstruction_ai4arctic.py configs/selfsup/ai4arctic/pretrain_80/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt80.py --checkpoint work_dirs/selfsup/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt80/iter_30000.pth --img-path "/home/jnoat92/scratch/dataset/ai4arctic/down_scale_9X/S1A_EW_GRDM_1SDH_20200424T101936_20200424T102036_032268_03BBA9_1CA8_icechart_cis_SGRDINFLD_20200424T1020Z_pl_a/00005.pkl" --out-file "work_dirs/selfsup/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt80/visual_reconstruction"