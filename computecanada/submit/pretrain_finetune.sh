#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=200G
#SBATCH --time=19:59:00
#SBATCH --output=../output/%j.out
#SBATCH --account=rrg-dclausi
#SBATCH --mail-user=jnoat92@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
set -e

module purge
module load  StdEnv/2020 python/3.10.2
module load gcc/9.3.0 opencv/4.8.0 cuda/11.7
echo "loading module done"

# cd /home/m32patel/projects/def-y2863che/ai4arctic/m32patel/mmselfsup

# echo "starting pretrain and finetune..."

# echo "Config file: $1"
# # srun --ntasks=2 --gres=gpu:2  --kill-on-bad-exit=1 --cpus-per-task=12 python tools/train.py $1 --launcher slurm --resume
# # srun --ntasks=2 --gres=gpu:2  --kill-on-bad-exit=1 --cpus-per-task=12 python tools/train.py $1 --launcher slurm
# srun --ntasks=2 --gres=gpu:2 --kill-on-bad-exit=1 --cpus-per-task=12 python tools/train.py $1 --launcher slurm

# # Extract the base name without extension
# base_name=$(basename "$1" .py)
# CHECKPOINT=$(cat work_dirs/selfsup/$base_name/last_checkpoint)
# echo "mmpretrain Checkpoint $CHECKPOINT"

# # train file 1
# python tools/analysis_tools/visualize_reconstruction.py $1  --use-vis-pipeline --checkpoint $CHECKPOINT --img-path "/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/S1B_EW_GRDM_1SDH_20211119T080313_20211119T080413_029654_0389FB_9317_icechart_dmi_202111190805_CentralEast_RIC.nc"  --out-file "work_dirs/selfsup/$base_name/20211117T212054"

# # train file 2
# python tools/analysis_tools/visualize_reconstruction.py $1  --use-vis-pipeline --checkpoint $CHECKPOINT --img-path "/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/S1B_EW_GRDM_1SDH_20191018T130839_20191018T130933_018530_022EA4_4267_icechart_cis_SGRDIMID_20191018T1302Z_pl_a.nc"  --out-file "work_dirs/selfsup/$base_name/20191018T130839"

# # test file 1
# python tools/analysis_tools/visualize_reconstruction.py $1  --use-vis-pipeline --checkpoint $CHECKPOINT --img-path "/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3/S1B_EW_GRDM_1SDH_20210328T202742_20210328T202842_026220_032117_57E3_icechart_dmi_202103282025_CapeFarewell_RIC.nc"  --out-file "work_dirs/selfsup/$base_name/20210328T202742"

# # test file 2 
# python tools/analysis_tools/visualize_reconstruction.py $1  --use-vis-pipeline --checkpoint $CHECKPOINT --img-path "/home/m32patel/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3/S1A_EW_GRDM_1SDH_20210430T205436_20210430T205537_037685_047252_CBB0_icechart_dmi_202104302055_SouthWest_RIC.nc"  --out-file "work_dirs/selfsup/$base_name/20210430T205436"

# deactivate

cd /home/jnoat92/projects/rrg-dclausi/ai4arctic/sea-ice-mmseg

source ~/env_mmsegmentation/bin/activate
echo "Activating virtual environment done"

export WANDB_MODE=offline
export WANDB_DATA_DIR='/home/jnoat92/scratch/wandb'

echo "Config file: $2"

# CHECKPOINT=(/home/jnoat92/projects/rrg-dclausi/ai4arctic/sea-ice-mmseg/work_dirs/mae_ai4arctic_ds2_pt_80_ft_20/iter_40000.pth)
# srun --ntasks=1 --gres=gpu:1  --kill-on-bad-exit=1 --cpus-per-task=8 python tools/train.py $2 --launcher none --cfg-options model.backbone.init_cfg.checkpoint=${CHECKPOINT} model.backbone.init_cfg.type='Pretrained' model.backbone.init_cfg.prefix='backbone.'

srun --ntasks=1 --gres=gpu:1  --kill-on-bad-exit=1 --cpus-per-task=8 python tools/train.py $2

# srun --ntasks=1 --gres=gpu:1  --kill-on-bad-exit=1 --cpus-per-task=8 python tools/train.py $2 --resume

# Extract the base name without extension
base_name_mmseg=$(basename "$2" .py)
CHECKPOINT_mmseg=$(cat work_dirs/$base_name_mmseg/last_checkpoint)
echo "mmseg checkpoint $CHECKPOINT_mmseg"

python tools/test.py $2 $CHECKPOINT_mmseg --out work_dirs/$base_name_mmseg/ --show-dir work_dirs/$base_name_mmseg/