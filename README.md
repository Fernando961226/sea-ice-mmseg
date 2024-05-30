# AI4arctic readme
 
This repository builds on top of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), a segmentation toolbox based on PyTorch. The mmsegmentation toolbox is particularly useful for quickly trying out different segmentation methods, as it includes implementations of many known architectures. However, the original repository is designed for general computer vision tasks and is not tailored for remote sensing or other specific types of data.

This repository extends mmsegmentation to support the use of remote sensing data for segmentation tasks. Additionally, for Arctic research, the repository includes a multi-task feature where the model has to predict three segmentation maps instead of one. This involves creating a multi-task model with three separate decoders, each dedicated to a specific segmentation task.

## Getting Started:

### Installation
Install the required packages and dependencies by running
```linux
bash compute_canada/submit/create_env.sh <env_name>
```
This will create a virtualenv at the location \~/<env_name>. (\~ stands for root/home folder)

### Creating config file

OpenMMLAB uses config file based experiments. This is very useful in Deep learning experiments where there are 100's of parameters and only few change at a time. Check out [this documentation](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html) for more information.

An example config file for ViT architecrure(obtained from MAE pretraining) with UPerNet architecture is given at [configs/multi_task_ai4arctic/mae_ai4arctic_ds5_pt_80_ft_20.py](configs/multi_task_ai4arctic/mae_ai4arctic_ds5_pt_80_ft_20.py)

### Submitting a job

To enable submitting multiple jobs easier on slurm (especially when the user wants to test different configurations), a shell script is created where the user only has to put the path of the config files in a list and they all are submitted.

```Shell
#!/bin/bash 
set -e
mmselfsup_config=( 
configs/selfsup/mask_ratio/mae_ai4arctic_ds5_pt_80_ft_20_mr50.py
configs/selfsup/mask_ratio/mae_ai4arctic_ds5_pt_80_ft_20_mr90.py
configs/selfsup/mask_ratio/mae_ai4arctic_ds5_pt_80_ft_20_mr25.py
)

mmseg_config=(
configs/mask_ratio/mae_ai4arctic_ds5_pt_80_ft_20_mr50.py
configs/mask_ratio/mae_ai4arctic_ds5_pt_80_ft_20_mr90.py
configs/mask_ratio/mae_ai4arctic_ds5_pt_80_ft_20_mr25.py
)

for i in "${!mmseg_config[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch pretrain_finetune.sh ${mmselfsup_config[i]} ${mmseg_config[i]}
   # bash test2.sh ${array[i]}
   # echo  ${array[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 5
done
```
To run the training script, run the below command
```Linux
bash computecanada/submit/submit_loop.sh
```

## The below section highlights the changes done to original mmsegmentation to get it working for ai4arctic with multi-task

### Template Configuration file:
[configs/multi_task_ai4arctic/mae_ai4arctic_ds5_pt_80_ft_20.py](configs/multi_task_ai4arctic/mae_ai4arctic_ds5_pt_80_ft_20.py)

### Dataset:
- Added a new dataset class `AI4Arctic`
- files added/modified: <br>
[mmseg/datasets/ai4arctic.py](mmseg/datasets/ai4arctic.py)

### Pipelines/image loader:

- Added a new tranform function `PreLoadImageandSegFromNetCDFFile`
- files added/modified: <br>
[mmseg/datasets/transforms/loading_ai4arctic.py](mmseg/datasets/transforms/loading_ai4arctic.py)
- Notes:
-- Loads the GT as 3 channel tensor instead of 1 channel(multiple GT for different tasks)

-- TODO:
1. Upscaling of low res variables
2. Add time, location encoding


### Model (To support multitask):

- Added a new `EncoderDecoder` class called `MultitaskEncoderDecoder` which takes 3 decoder dictionary as input.

eg in config file

```python
decode_head=[
    dict(
        type='UPerHead',
        task='SIC',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    dict(
        type='UPerHead',
        task='SOD',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    dict(
        type='UPerHead',
        task='FLOE',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
],
```

- Modifed class `BaseDecodeHead` to pass the task number, this is required to disitinguish which Ground truth is used to calculate loss.
- [Difference] (https://git.uwaterloo.ca/vip_ai4arctic/mmsegmentation/-/commit/9b4ea6cd9a8a8e93edece0825f71f47f13f0f9d9#669e3eb0aa8bdb6592e42e25e11896ff7c8a2123)
### Metric
- Added a new metric class `MultitaskIoUMetric`
- files added/modified:
[mmseg/evaluation/metrics/multitask_iou_metric.py](mmseg/evaluation/metrics/multitask_iou_metric.py)

### Visualization/ Custom metric Hook

- Added a new hook called `SegAI4ArcticVisualizationHook`
- files added/modified:
[mmseg/engine/hooks/ai4arctic_visualization_hook.py](mmseg/engine/hooks/ai4arctic_visualization_hook.py)
- Notes:
This hook is responsible for calculating R2/F1/Combined score as well as plotting the prediction and saving them to a folder