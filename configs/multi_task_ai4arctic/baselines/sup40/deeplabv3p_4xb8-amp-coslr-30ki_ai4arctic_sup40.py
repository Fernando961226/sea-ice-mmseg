_base_ = [
    '../../vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft40.py'
]

# ============== MODEL ==============
# DeepLabv3p from --> configs/_base_/models/deeplabv3plus_r50-d8.py
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNetV1c',
        depth=50,
        in_channels=len(_base_.channels),
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck = None,
    decode_head = [
        dict(type='DepthwiseSeparableASPPHead_regression',
            in_channels=2048,
            in_index=3,
            channels=512,
            dilations=(1, 12, 24, 36),
            c1_in_channels=256,
            c1_channels=48,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            num_classes=11,
            task='SIC',
            loss_decode=dict(
                type='MSELossWithIgnoreIndex', loss_weight=1.0)),
        dict(type='DepthwiseSeparableASPPHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            dilations=(1, 12, 24, 36),
            c1_in_channels=256,
            c1_channels=48,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            num_classes=6,
            task='SOD',
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=3.0, avg_non_ignore=False)),
        dict(type='DepthwiseSeparableASPPHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            dilations=(1, 12, 24, 36),
            c1_in_channels=256,
            c1_channels=48,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            num_classes=7,
            task='FLOE',
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=3.0, avg_non_ignore=False)),
    ],
    )

wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)



