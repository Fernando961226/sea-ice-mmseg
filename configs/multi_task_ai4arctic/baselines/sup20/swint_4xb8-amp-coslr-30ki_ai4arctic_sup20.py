_base_ = [
    '../../vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20.py'
]

# ============== MODEL ==============
# Swin-base from --> configs/swin/swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py

decode_head = _base_.model.decode_head
for i in range(len(decode_head)):
    decode_head[i]['in_channels'] = [128, 256, 512, 1024]
    decode_head[i]['channels'] = 512

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=_base_.crop_size,
        in_channels=len(_base_.channels),
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32]
        ),
    neck=None,
    decode_head=decode_head,
    )

wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)


