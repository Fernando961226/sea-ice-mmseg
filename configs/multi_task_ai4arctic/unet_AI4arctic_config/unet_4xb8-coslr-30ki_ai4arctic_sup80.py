'''
No@
'''
_base_ = [
    '../vit/mae_vit-base_4xb8-coslr-30ki_ai4arctic_ft80.py'
]

# ============== MODEL ==============
data_preprocessor = dict(test_cfg=dict(size_divisor=16))    # test_cfg into data_preprocessor provides 
                                                            # automatic padding required for predictions in mode 'whole'
model = dict(
    type='MultitaskEncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='/project/6075102/AI4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_vit-base-p16/epoch_200.pth',
    # pretrained=None,
    backbone=dict(
        _delete_=True,
        type='AI4Arctic_UNet',
        # pretrained='/home/m32patel/projects/def-dclausi/AI4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_vit-base-p16_cs512-amp-coslr-400e_ai4arctic_norm_pix/epoch_400.pth',
        # pretrained='/project/6075102/AI4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_vit-base-p16/epoch_200.pth',
        # init_cfg=dict(type='Pretrained', checkpoint=None, prefix = 'backbone.'),
        in_channels=len(_base_.channels),
        # base_channels=32,
        layer_channels = [32, 64, 64, 64, 64],
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=_base_.norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='DeconvModule', kernel_size=2),
        norm_eval=False,
        dcn=None,
        plugins=None,
        pretrained=None,
        init_cfg=None),
    neck=None,
    decode_head=[
        dict(
            # type='FCNHead',
            type='FCNHead_regression',
            task='SIC',
            num_classes=11,

            num_convs=0,
            concat_input=False,
            in_channels=32,
            in_index=-1,
            channels=32,
            dropout_ratio=0,
            norm_cfg=_base_.norm_cfg,
            align_corners=False,
            loss_decode=dict(
                # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=False)),
                type='MSELossWithIgnoreIndex', loss_weight=1.0)),
        dict(
            type='FCNHead',
            task='SOD',
            num_classes=6,

            num_convs=0,
            concat_input=False,
            in_channels=32,
            in_index=-1,
            channels=32,
            dropout_ratio=0,
            norm_cfg=_base_.norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=3.0, avg_non_ignore=False)),
        dict(
            type='FCNHead',
            task='FLOE',
            num_classes=7,

            num_convs=0,
            concat_input=False,
            in_channels=32,
            in_index=-1,
            channels=32,
            dropout_ratio=0,
            norm_cfg=_base_.norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=3.0, avg_non_ignore=False)),
    ],
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0] *66//100, crop_size[1]*66//100)))


wandb_config = _base_.wandb_config
wandb_config.init_kwargs.name = '{{fileBasenameNoExtension}}'
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)

custom_imports = _base_.custom_imports
custom_imports.imports.extend([
                                'mmseg.models.backbones.ai4arctic_unet',
                                ])

