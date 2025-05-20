#!/bin/bash 
set -e
# # Supervised
from_scratch_configs=( 

# ========== ViT
# # base
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_sup20.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_sup40.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_sup60.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_sup80.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_sup100.py

# # large
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_sup20.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_sup40.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_sup60.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_sup80.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_sup100.py

# # huge
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_sup20.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_sup40.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_sup60.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_sup80.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_sup100.py

# ========== Baselines

# # sup20
# configs/multi_task_ai4arctic/baselines/sup20/convnext_4xb8-amp-coslr-30ki_ai4arctic_sup20.py
# configs/multi_task_ai4arctic/baselines/sup20/deeplabv3p_4xb8-amp-coslr-30ki_ai4arctic_sup20.py
# configs/multi_task_ai4arctic/baselines/sup20/resnet50_4xb8-amp-coslr-30ki_ai4arctic_sup20.py
# configs/multi_task_ai4arctic/baselines/sup20/segformer_4xb8-amp-coslr-30ki_ai4arctic_sup20.py
# configs/multi_task_ai4arctic/baselines/sup20/swint_4xb8-amp-coslr-30ki_ai4arctic_sup20.py
# # configs/multi_task_ai4arctic/baselines/sup20/unet_4xb8-amp-coslr-30ki_ai4arctic_sup20.py

# # sup40
# configs/multi_task_ai4arctic/baselines/sup40/convnext_4xb8-amp-coslr-30ki_ai4arctic_sup40.py
# configs/multi_task_ai4arctic/baselines/sup40/deeplabv3p_4xb8-amp-coslr-30ki_ai4arctic_sup40.py
# configs/multi_task_ai4arctic/baselines/sup40/resnet50_4xb8-amp-coslr-30ki_ai4arctic_sup40.py
# configs/multi_task_ai4arctic/baselines/sup40/segformer_4xb8-amp-coslr-30ki_ai4arctic_sup40.py
# configs/multi_task_ai4arctic/baselines/sup40/swint_4xb8-amp-coslr-30ki_ai4arctic_sup40.py
# configs/multi_task_ai4arctic/baselines/sup40/unet_4xb8-amp-coslr-30ki_ai4arctic_sup40.py

# # sup60
# configs/multi_task_ai4arctic/baselines/sup60/convnext_4xb8-amp-coslr-30ki_ai4arctic_sup60.py
# configs/multi_task_ai4arctic/baselines/sup60/deeplabv3p_4xb8-amp-coslr-30ki_ai4arctic_sup60.py
# configs/multi_task_ai4arctic/baselines/sup60/resnet50_4xb8-amp-coslr-30ki_ai4arctic_sup60.py
# configs/multi_task_ai4arctic/baselines/sup60/segformer_4xb8-amp-coslr-30ki_ai4arctic_sup60.py
# configs/multi_task_ai4arctic/baselines/sup60/swint_4xb8-amp-coslr-30ki_ai4arctic_sup60.py
# configs/multi_task_ai4arctic/baselines/sup60/unet_4xb8-amp-coslr-30ki_ai4arctic_sup60.py

# # sup80
# configs/multi_task_ai4arctic/baselines/sup80/convnext_4xb8-amp-coslr-30ki_ai4arctic_sup80.py
# configs/multi_task_ai4arctic/baselines/sup80/deeplabv3p_4xb8-amp-coslr-30ki_ai4arctic_sup80.py
# configs/multi_task_ai4arctic/baselines/sup80/resnet50_4xb8-amp-coslr-30ki_ai4arctic_sup80.py
# configs/multi_task_ai4arctic/baselines/sup80/segformer_4xb8-amp-coslr-30ki_ai4arctic_sup80.py
# configs/multi_task_ai4arctic/baselines/sup80/swint_4xb8-amp-coslr-30ki_ai4arctic_sup80.py
# configs/multi_task_ai4arctic/baselines/sup80/unet_4xb8-amp-coslr-30ki_ai4arctic_sup80.py

# # sup100
# configs/multi_task_ai4arctic/baselines/sup100/convnext_4xb8-amp-coslr-30ki_ai4arctic_sup100.py
# configs/multi_task_ai4arctic/baselines/sup100/deeplabv3p_4xb8-amp-coslr-30ki_ai4arctic_sup100.py
# configs/multi_task_ai4arctic/baselines/sup100/resnet50_4xb8-amp-coslr-30ki_ai4arctic_sup100.py
# configs/multi_task_ai4arctic/baselines/sup100/segformer_4xb8-amp-coslr-30ki_ai4arctic_sup100.py
# configs/multi_task_ai4arctic/baselines/sup100/swint_4xb8-amp-coslr-30ki_ai4arctic_sup100.py
# configs/multi_task_ai4arctic/baselines/sup100/unet_4xb8-amp-coslr-30ki_ai4arctic_sup100.py

)

for i in "${!from_scratch_configs[@]}"; do
   sbatch from_scratch.sh ${from_scratch_configs[i]}
   echo "task successfully submitted" 
   sleep 2
done
