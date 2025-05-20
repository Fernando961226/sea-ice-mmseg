'''
No@
'''
_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]

# ============== DATASET ==============
import os
import numpy as np

crop_size = (256, 256)
downsample_factor_train = [5]   # List all downsampling factors from 2X to 10X to include during training
downsample_factor_test = 5

GT_type = ['SIC', 'SOD', 'FLOE']
combined_score_weights = [2/5, 2/5, 1/5]

# dataset settings
dataset_type_train = 'AI4ArcticPatches'
dataset_type_val = 'AI4Arctic'

data_root_train_nc = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3'
gt_root_train = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3_segmaps'
data_root_test_nc = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3'
gt_root_test = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3_segmaps'
data_root_patches = '/home/jnoat92/scratch/dataset/ai4arctic/'

file_train = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/data_split_setup/train_100.txt'
file_val = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/data_split_setup/val_file.txt'
file_test = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/data_split_setup/test_file.txt'

# # small data to test
# data_root_test_nc = data_root_train_nc
# gt_root_test = gt_root_train
# file_test = '/home/jnoat92/projects/rrg-dclausi/ai4arctic/dataset/data_split_setup/t_1.txt'
# file_train = file_test; file_val = file_test

# load normalization params
possible_channels = ['nersc_sar_primary', 'nersc_sar_secondary', 
                     'distance_map', 
                     'btemp_6_9h', 'btemp_6_9v', 'btemp_7_3h', 'btemp_7_3v', 'btemp_10_7h', 'btemp_10_7v', 'btemp_18_7h',
                     'btemp_18_7v', 'btemp_23_8h', 'btemp_23_8v', 'btemp_36_5h', 'btemp_36_5v', 'btemp_89_0h', 'btemp_89_0v',
                     'u10m_rotated', 'v10m_rotated', 't2m', 'skt', 'tcwv', 'tclw', 
                     'sar_grid_incidenceangle', 
                     'sar_grid_latitude', 'sar_grid_longitude', 'month', 'day']
global_meanstd = np.load(os.path.join(data_root_train_nc, 'global_meanstd.npy'), allow_pickle=True).item()
mean, std = {}, {}
for i in possible_channels:
    ch = i if i != 'sar_grid_incidenceangle' else 'sar_incidenceangle'
    if ch not in global_meanstd.keys(): continue
    mean[i] = global_meanstd[ch]['mean']
    std[i]  = global_meanstd[ch]['std']

mean['sar_grid_latitude'] = 69.14857395508363;   std['sar_grid_latitude']  = 7.023603113019076
mean['sar_grid_longitude']= -56.351130746236606; std['sar_grid_longitude'] = 31.263271402859893
mean['month'] = 6; std['month']  = 3.245930125274979
mean['day'] = 182; std['day']  = 99.55635507719892


# channels to use
channels = [
    # -- Sentinel-1 variables -- #
    'nersc_sar_primary',
    'nersc_sar_secondary',

    # # -- incidence angle -- #
    # 'sar_grid_incidenceangle',

    # # -- Geographical variables -- #
    'sar_grid_latitude',
    'sar_grid_longitude',
    # 'distance_map',

    # # # -- AMSR2 channels -- #
    # 'btemp_6_9h', 'btemp_6_9v',
    # 'btemp_7_3h', 'btemp_7_3v',
    # 'btemp_10_7h', 'btemp_10_7v',
    'btemp_18_7h', 'btemp_18_7v',
    # 'btemp_23_8h', 'btemp_23_8v',
    'btemp_36_5h', 'btemp_36_5v',
    # 'btemp_89_0h', 'btemp_89_0v',

    # # # -- Environmental variables -- #
    'u10m_rotated', 'v10m_rotated',
    't2m', 
    # 'skt', 
    'tcwv', 'tclw',

    # # -- acquisition time
    'month', 
    # 'day'
]


# ------------- TRAIN SETUP
train_pipeline = [
    dict(type='LoadPatchFromPKLFile', channels=channels, mean=mean, std=std, 
         to_float32=True, nan=255, with_seg=True, GT_type=GT_type),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(1.0, 1.5),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.9),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion')
    dict(type='PackSegInputs')
]

concat_dataset = dict(type='ConcatDataset', 
                      datasets= [dict(type=dataset_type_train,
                                      data_root = os.path.join(data_root_patches, 'down_scale_%dX'%(i)),
                                      ann_file = file_train,
                                      pipeline = train_pipeline) for i in downsample_factor_train])
train_dataloader = dict(batch_size=16,
                        num_workers=8,
                        persistent_workers=True,
                        sampler=dict(type='WeightedInfiniteSampler', use_weights=True),
                        # sampler=dict(type='InfiniteSampler', shuffle=True),
                        dataset=concat_dataset)

# ------------- VAL SETUP
val_pipeline = [
    dict(type='PreLoadImageandSegFromNetCDFFile', data_root=data_root_train_nc, gt_root=gt_root_train, 
         ann_file=file_val, channels=channels, mean=mean, std=std, to_float32=True, nan=255, 
         downsample_factor=downsample_factor_test, with_seg=True, GT_type=GT_type),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                                          'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                          'flip_direction', 'reduce_zero_label', 'dws_factor')) 
                                          # 'dws_factor' is the only non-default parameter
]
val_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type_val,
                                   data_root=data_root_train_nc,
                                   ann_file=file_val,
                                   pipeline=val_pipeline))

# ------------- TEST SETUP
test_pipeline = [
    dict(type='PreLoadImageandSegFromNetCDFFile', data_root=data_root_test_nc, gt_root=gt_root_test, 
         ann_file=file_test, channels=channels, mean=mean, std=std, to_float32=True, nan=255, 
         downsample_factor=downsample_factor_test, with_seg=True, GT_type=GT_type),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                                          'img_shape', 'pad_shape', 'scale_factor', 'flip',
                                          'flip_direction', 'reduce_zero_label', 'dws_factor')) 
                                            # 'dws_factor' is the only non-default parameter, 
                                            # I need it in the visualization hook
]
test_dataloader = dict(batch_size=1,
                      num_workers=4,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type_val,
                                   data_root=data_root_test_nc,
                                   ann_file=file_test,
                                   pipeline=test_pipeline))


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=None,
    std=None,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=16)) # test_cfg into data_preprocessor provides 
                                    # automatic padding required for predictions in mode 'whole'
model = dict(
    type='MultitaskEncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='/project/6075102/AI4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_vit-base-p16/epoch_200.pth',
    # pretrained=None,
    backbone=dict(
        type='AI4Arctic_UNet',
        # pretrained='/home/m32patel/projects/def-dclausi/AI4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_vit-base-p16_cs512-amp-coslr-400e_ai4arctic_norm_pix/epoch_400.pth',
        # pretrained='/project/6075102/AI4arctic/m32patel/mmselfsup/work_dirs/selfsup/mae_vit-base-p16/epoch_200.pth',
        # init_cfg=dict(type='Pretrained', checkpoint=None, prefix = 'backbone.'),
        in_channels=len(channels),
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
        norm_cfg=norm_cfg,
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
            norm_cfg=norm_cfg,
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
            norm_cfg=norm_cfg,
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
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=3.0, avg_non_ignore=False)),
    ],
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0] *66//100, crop_size[1]*66//100)))


val_evaluator = dict(type='MultitaskAi4arcticMetric', tasks=GT_type, 
                     custom_metrics={'SIC': ['r2', 'mIoU'], 
                                     'SOD': ['f1', 'mIoU'], 
                                     'FLOE': ['f1', 'mIoU']}, 
                     combined_score_weights = dict(zip(GT_type, combined_score_weights)),
                     num_classes = {'SIC': 11, 'SOD': 6, 'FLOE': 7})
test_evaluator = val_evaluator

# ============== SCHEDULE ==============
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, weight_decay=0.01, momentum=0.9))

n_iterations = 150000
val_interval = 500
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=n_iterations, val_interval=val_interval)

# learning rate scheduler
# param_scheduler = [dict(type='CosineAnnealingLR', T_max=10000, by_epoch=False) for i in range(n_iterations//10000)]
param_scheduler = [ dict(type='CosineRestartLR',
                         periods=[10000 for i in range(n_iterations//10000)],
                         restart_weights = [1] * (n_iterations//10000),
                         eta_min=1e-6,
                         by_epoch=False)]


# ============== RUNTIME ==============
metrics = {'SIC': 'r2', 'SOD': 'f1', 'FLOE': 'f1'}
num_classes = {'SIC': 12, 'SOD': 7, 'FLOE': 8} # add 1 class extra for visualization to work correctly, put [11,6,7] in other places
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='AI4arcticLoggerHook', interval=val_interval//5, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', 
                    # save_best="combined_score", 
                    save_best=["combined_score", "SIC.r2", "SOD.f1", "FLOE.f1"], 
                    rule="greater",
                    by_epoch=False, 
                    interval=-1, save_last=True,
                    max_keep_ckpts=2),
    # early_stopping=dict(type='EarlyStoppingHookMain', 
    #                 monitor="combined_score", rule="greater",
    #                 min_delta=0.0, patience=15),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegAI4ArcticVisualizationHook', 
                       tasks=GT_type, num_classes=num_classes, 
                       downsample_factor=None, metrics=metrics, 
                       combined_score_weights=combined_score_weights, 
                       draw=True),
    runtime_info=dict(type='AI4arcticRuntimeInfoHook')
    )

log_processor = dict(type='LogProcessor', log_with_hierarchy=True)  # log_with_hierarchy allows separating metrics 
                                                                    # (train-val-test) in loggers like Tensorboard or Wandb
wandb_config = dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         entity='jnoat92',
                         project='Ai4arctic_config',
                         name='{{fileBasenameNoExtension}}',),
                     #  name='filename',),
                     define_metric_cfg=None,
                     commit=True,
                     log_code_name=None,
                     watch_kwargs=None)
vis_backends = [wandb_config, dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)


custom_imports = dict(
    imports=['mmseg.datasets.ai4arctic_patches',
             'mmseg.datasets.transforms.loading_ai4arctic_patches',
             'mmseg.structures.sampler.ai4arctic_multires_sampler',

             'mmseg.models.segmentors.mutitask_encoder_decoder',
             'mmseg.models.backbones.ai4arctic_unet',
             'mmseg.models.losses.mse_loss',

             'mmseg.evaluation.metrics.multitask_ai4arctic_metric',

             'mmseg.engine.hooks.ai4arctic_visualization_hook',
             'mmseg.engine.hooks.early_stopping_hook_main',
             'mmseg.engine.hooks.ai4arctic_runtime_hook',
             'mmseg.engine.hooks.ai4arctic_logger_hook'],
    allow_failed_imports=False)

# randomness
randomness = dict(seed=0, diff_rank_seed=True)