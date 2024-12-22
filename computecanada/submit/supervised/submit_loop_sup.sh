#!/bin/bash 
set -e
supervised_configs=( 
# AI4arctic config
configs/multi_task_ai4arctic/unet_ai4arctic_patches_ds5X_pt_80_ft_20_2.py

# # VIT
# # base
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft80.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft60.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft40.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20.py

# # large
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft80.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft60.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft40.py   
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft20.py

# # huge
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft80.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft60.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft40.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft20.py
)
for i in "${!supervised_configs[@]}"; do
   sbatch sup.sh ${supervised_configs[i]}
   echo "task successfully submitted" 
   sleep 2
done
