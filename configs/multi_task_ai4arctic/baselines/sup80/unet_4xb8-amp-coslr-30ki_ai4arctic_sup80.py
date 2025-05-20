_base_ = [
    '../../vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft80.py'
]

# # ==================== SMALL DATASET SCRIPT
# data_root_test_nc = _base_.data_root_train_nc
# file_val = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/data_split_setup/t_1.txt'
# file_train = file_val; file_test = file_val

# concat_dataset = _base_.concat_dataset
# train_dataloader = _base_.train_dataloader
# for i in range(len(concat_dataset.datasets)):
#     concat_dataset.datasets[i].ann_file = file_train
# train_dataloader.dataset = concat_dataset

# val_pipeline = _base_.val_pipeline
# val_pipeline[0].ann_file=file_val
# val_dataloader = _base_.val_dataloader
# val_dataloader.dataset.ann_file = file_val
# val_dataloader.dataset.pipeline = val_pipeline

# test_pipeline = _base_.test_pipeline
# test_pipeline[0].ann_file=file_test
# test_dataloader = _base_.test_dataloader
# test_dataloader.dataset.ann_file = file_test
# test_dataloader.dataset.pipeline = test_pipeline
# # ==================== 


# ============== MODEL ==============
data_preprocessor = dict(test_cfg=dict(size_divisor=16))    # test_cfg into data_preprocessor provides 
                                                            # automatic padding required for predictions in mode 'whole'
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='AI4Arctic_UNet',
        in_channels=len(_base_.channels),
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
