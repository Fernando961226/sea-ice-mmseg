_base_ = [
    '../../vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft60.py'
]

# ============== MODEL ==============
# ResNet50 from --> configs/_base_/models/upernet_r50.py
decode_head = _base_.model.decode_head
for i in range(len(decode_head)):
    decode_head[i]['in_channels'] = [256, 512, 1024, 2048]
    # decode_head[i]['in_channels'] = [64, 128, 256, 512]
    decode_head[i]['channels'] = 512

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNetV1c',
        depth=50,
        in_channels=len(_base_.channels),
        out_indices=(0, 1, 2, 3),
        norm_cfg=_base_.norm_cfg,
        contract_dilation=True),
    neck = None,
    decode_head = decode_head,
    )


wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)