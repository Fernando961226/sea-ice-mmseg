#!/bin/bash 
set -e
supervised_configs=( 
# AI4arctic config
# configs/multi_task_ai4arctic/unet_AI4arctic_config/unet_ai4arctic_patches_ds5X.py
configs/multi_task_ai4arctic/unet_AI4arctic_config/unet_ai4arctic_patches_ds5X_sup20.py

)
for i in "${!supervised_configs[@]}"; do
   sbatch sup.sh ${supervised_configs[i]}
   echo "task successfully submitted" 
   sleep 2
done
