_base_ = [
    '../../vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20.py'
]

# ============== MODEL ==============
# convext (base) from --> configs/_base_/models/upernet_convnext.py
data_preprocessor = dict(test_cfg=dict(size_divisor=32))    # test_cfg into data_preprocessor provides 
                                                            # automatic padding required for predictions in mode 'whole'

decode_head = _base_.model.decode_head
for i in range(len(decode_head)):
    decode_head[i]['in_channels'] = [128, 256, 512, 1024]
    decode_head[i]['channels'] = 512

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='base',
        in_channels=len(_base_.channels),
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False),
    neck = None,
    decode_head = decode_head,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(_delete_=True, mode='whole')
    )


wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)



