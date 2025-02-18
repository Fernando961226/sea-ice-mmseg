#!/bin/bash 
set -e
selfsup_configs=( 
# # base
# configs/selfsup/ai4arctic/pretrain_20/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt20.py
# configs/selfsup/ai4arctic/pretrain_40/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt40.py
# configs/selfsup/ai4arctic/pretrain_60/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt60.py
# configs/selfsup/ai4arctic/pretrain_80/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt80.py

# configs/selfsup/ai4arctic/pretrain_80/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt80_mr70.py
# configs/selfsup/ai4arctic/pretrain_80/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt80_mr75.py
# configs/selfsup/ai4arctic/pretrain_80/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt80_mr80.py
configs/selfsup/ai4arctic/pretrain_80/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt80_mr85.py
configs/selfsup/ai4arctic/pretrain_80/mae_vit-base-p16_4xb8-amp-coslr-50ki_ai4arctic_pt80_mr90.py

# # large
# configs/selfsup/ai4arctic/pretrain_20/mae_vit-large-p16_4xb8-amp-coslr-50ki_ai4arctic_pt20.py
# configs/selfsup/ai4arctic/pretrain_40/mae_vit-large-p16_4xb8-amp-coslr-50ki_ai4arctic_pt40.py
# configs/selfsup/ai4arctic/pretrain_60/mae_vit-large-p16_4xb8-amp-coslr-50ki_ai4arctic_pt60.py
# configs/selfsup/ai4arctic/pretrain_80/mae_vit-large-p16_4xb8-amp-coslr-50ki_ai4arctic_pt80.py

# # huge
# configs/selfsup/ai4arctic/pretrain_20/mae_vit-huge-p16_4xb8-amp-coslr-50ki_ai4arctic_pt20.py
# configs/selfsup/ai4arctic/pretrain_40/mae_vit-huge-p16_4xb8-amp-coslr-50ki_ai4arctic_pt40.py
# configs/selfsup/ai4arctic/pretrain_60/mae_vit-huge-p16_4xb8-amp-coslr-50ki_ai4arctic_pt60.py
# configs/selfsup/ai4arctic/pretrain_80/mae_vit-huge-p16_4xb8-amp-coslr-50ki_ai4arctic_pt80.py

)

fintune_configs=( 
# # base
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft80.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft60.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft40.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20.py


# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr70.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr75.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr80.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr85.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr90.py

configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_sup20.py
configs/multi_task_ai4arctic/unet_AI4arctic_config/unet_4xb8-amp-coslr-30ki_ai4arctic_sup20.py

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


for i in "${!selfsup_configs[@]}"; do
   sbatch finetune.sh ${selfsup_configs[i]} ${fintune_configs[i]}
   echo "task successfully submitted" 
   sleep 2
done

# supervised_configs=( 
# configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-300e_ai4arctic.py
# configs/selfsup/ai4arctic/mae_vit-base-p16_8xb512-amp-coslr-400e_ai4arctic.py
# configs/multi_task_ai4arctic/unet_ai4arctic_patches_ds5X_pt_80_ft_20_2.py

# # Supervised
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft100.
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft100.py   
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft100.py 
# )
# for i in "${!supervised_configs[@]}"; do
#    sbatch finetune.sh $i ${supervised_configs[i]}
#    echo "task successfully submitted" 
#    sleep 2
# done
