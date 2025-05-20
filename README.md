# AI4arctic readme

This repository builds on top of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation.git). The mmsegmentation toolbox is particularly useful for quick semantic segmentation setup towards training and testing deep learning methods, as it includes implementations of many known architectures as well as the required pipeline to integrate data, models, and evaluation. The original repository is designed for general computer vision tasks and is not tailored for remote sensing or other specific types of data.

This repository extends mmsegmentation to support remote sensing data for segmentation tasks. Specifically, for Arctic research, the repository includes a multi-task feature where the model is able to predict three segmentation maps instead of one. This involves creating a multi-task segmentor with three separate decoders, each dedicated to a specific segmentation task.

## Changes done to original mmsegmentation to get it working for ai4arctic

### Datasets:
- `AI4Arctic`: Prepared to load all training images (downsampled) in the RAM.
- `AI4ArcticPatches`: Prepared to operate with pre-computed patches.
- files added/modified: <br>
[mmseg/datasets/ai4arctic_patches.py](mmseg/datasets/ai4arctic_patches.py), [mmseg/datasets/transforms/loading_ai4arctic_patches.py](mmseg/datasets/transforms/loading_ai4arctic_patches.py)

### Multi-task pipeline:
- Encoder-Decoder model to support SIC, SOD, and FLOE tasks: [mmseg/models/segmentors/mutitask_encoder_decoder.py](mmseg/models/segmentors/mutitask_encoder_decoder.py)
- `BaseDecodeHead` to support multitask models: [mmseg/models/decode_heads/decode_head_multitask.py](mmseg/models/decode_heads/decode_head_multitask.py)

### Regression target
- FCN head for regression: [FCNHead_regression](https://github.com/Fernando961226/sea-ice-mmseg/blob/e66a789fc8d7e5a320b39dccf748dd6965b668f4/mmseg/models/decode_heads/fcn_head.py#L104)
- [Mse_loss](mmseg/models/losses/mse_loss.py)
- UperNet head for regression: [UPerHead_regression](https://github.com/Fernando961226/sea-ice-mmseg/blob/e66a789fc8d7e5a320b39dccf748dd6965b668f4/mmseg/models/decode_heads/uper_head.py#L147C7-L147C26)

### Modified Hooks:
- [mmseg/engine/hooks/ai4arctic_checkpoint_hook.py](mmseg/engine/hooks/ai4arctic_checkpoint_hook.py)
- [mmseg/engine/hooks/ai4arctic_logger_hook.py](mmseg/engine/hooks/ai4arctic_logger_hook.py)
- [mmseg/engine/hooks/ai4arctic_runtime_hook.py](mmseg/engine/hooks/ai4arctic_runtime_hook.py)
- [mmseg/engine/hooks/ai4arctic_visualization_hook.py](mmseg/engine/hooks/ai4arctic_visualization_hook.py)
- [mmseg/engine/hooks/early_stopping_hook_main.py](mmseg/engine/hooks/early_stopping_hook_main.py)

### Metrics
- [mmseg/evaluation/metrics/multitask_ai4arctic_metric.py](mmseg/evaluation/metrics/multitask_ai4arctic_metric.py)
- [mmseg/evaluation/metrics/multitask_iou_metric.py](mmseg/evaluation/metrics/multitask_iou_metric.py)

### Models
- Winner solution from [Autoice Challenge Competition](https://ai4eo.eu/portfolio/autoice-challenge/): [mmseg/models/backbones/ai4arctic_unet.py](mmseg/models/backbones/ai4arctic_unet.py)
- [mmseg/models/backbones/custom_vit_bckbn.py](mmseg/models/backbones/custom_vit_bckbn.py)

### Config
Config files to run experiments are located at [configs/multi_task_ai4arctic/](configs/multi_task_ai4arctic/)

## How to use:
1. Follow steps on [create_env.sh](computecanada/submit/create_env.sh) to create python environment. Installation steps on file create_env.sh work properly in Compute Canada. In other platforms package conflicts may appear.
2. [extract_patches.sh](computecanada/submit/extract_patches.sh) precompute patches and save them in the scratch folder (select the desired downsampling ratio).
3. [submit_loop_from_scratch.sh](computecanada/submit/submit_loop_from_scratch.sh) shows how to iterate over different config files to train models from scratch.
4. [submit_loop_pretrain.sh](computecanada/submit/submit_loop_pretrain.sh) shows how to iterate over different config files to train self-supervised models on [sea-ice-mmselfsup](https://github.com/jnoat92/sea-ice-mmselfsup.git)
5. [submit_loop_finetune.sh](computecanada/submit/submit_loop_finetune.sh) shows how to iterate over different config files to finetune after pre-training is done.
