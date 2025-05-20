'''
No@
'''
_base_ = ['mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft60.py']

wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)

