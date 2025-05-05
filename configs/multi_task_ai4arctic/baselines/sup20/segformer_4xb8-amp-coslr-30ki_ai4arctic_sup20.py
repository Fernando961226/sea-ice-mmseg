_base_ = [
    '../../vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20.py'
]

# ============== MODEL ==============
# Segformer_b2 --> /configs/segformer/segformer_mit-b2_8xb2-160k_ade20k-512x512.py
data_preprocessor = dict(test_cfg=dict(size_divisor=32))    # test_cfg into data_preprocessor provides 
                                                            # automatic padding required for predictions in mode 'whole'

model = dict(
    backbone=dict(
        _delete_=True,
        type='MixVisionTransformer',
        in_channels=len(_base_.channels),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 6, 3]),
    neck=None,
    decode_head=[
        dict(
            type='SegformerHead_regression',
            task='SIC',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            num_classes=11,
            channels=256,
            loss_decode=dict(
                type='MSELossWithIgnoreIndex', loss_weight=1.0),
            ),
        dict(
            type='SegformerHead',
            task='SOD',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            num_classes=6,
            channels=256,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=3.0, avg_non_ignore=False),
            ),
        dict(
            type='SegformerHead',
            task='FLOE',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            num_classes=7,
            channels=256,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=3.0, avg_non_ignore=False)
            ),
    ],
    )


wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)



