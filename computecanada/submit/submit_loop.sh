#!/bin/bash 
set -e
array=( 
# configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic.py
# configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-400e_ai4arctic.py

# configs/multi_task_ai4arctic/unet_ai4arctic_patches_ds5X_pt_80_ft_20_2.py

# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds2X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds2X3X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds2X4X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds2X5X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds2X6X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds2X7X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds2X8X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds2X9X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds2X10X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds3X_pt_80_ft_20.py           #CUDA out of memory
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds3X4X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds3X5X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds3X6X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds3X7X_pt_80_ft_20.py           #CUDA out of memory
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds3X8X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds3X9X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds3X10X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds4X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds4X5X_pt_80_ft_20.py           #CUDA out of memory
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds4X6X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds4X7X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds4X8X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds4X9X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds4X10X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds5X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds5X6X_pt_80_ft_20.py           #CUDA out of memory
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds5X7X_pt_80_ft_20.py           #CUDA out of memory
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds5X8X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds5X9X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds5X10X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds6X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds6X7X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds6X8X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds6X9X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds6X10X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds7X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds7X8X_pt_80_ft_20.py           #CUDA out of memory

# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds7X9X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds7X10X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds8X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds8X9X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds8X10X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds9X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds9X10X_pt_80_ft_20.py
# configs/multi_task_ai4arctic/mae_ai4arctic_patches_ds10X_pt_80_ft_20.py
)
for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch pretrain_finetune.sh $i ${array[i]}
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 2
done
