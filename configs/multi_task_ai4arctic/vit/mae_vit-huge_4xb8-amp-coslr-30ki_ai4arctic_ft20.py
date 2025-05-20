'''
No@
'''
_base_ = ['mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20.py']

# ============== DATASET ==============
train_dataloader = dict(batch_size=8, num_workers=4)

# Update backbone
arch='h'
decode_head=_base_.model.decode_head
for i in range(len(decode_head)):
    decode_head[i]['in_channels'] = _base_.decode_in_channels[arch]

model = dict(
    backbone=dict(arch=arch, out_indices=_base_.out_indices[arch]),
    neck=dict(embed_dim=_base_.decode_in_channels[arch][0]),
    decode_head=decode_head
    )

wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)

