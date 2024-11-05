dataset_type = 'CocoDataset'
data_root = '/home/g-zhu/sora/mmdetection_abate/mmdetection/coco_customized/'
metainfo = {
    'classes': ['Dolphin', 'Elephant', 'Guitar', 'Piano', 'Rabbit', 'Violin']
}
NUM_CLASSES = len(metainfo['classes'])

# RTMDet-tiny
# resume = True
# resume_from = '/home/g-zhu/sora/mmdetection_abate/mmdetection/work_dirs/rtmdet_l_8xb32-300e_coco_test/epoch_31.pth'
backbone_pretrain = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
deepen_factor = 0.167
widen_factor = 0.375
in_channels = [96, 192, 384]
neck_out_channels = 96
num_csp_blocks = 1
exp_on_reg = False

MAX_EPOCHS = 200
TRAIN_BATCH_SIZE = 17
VAL_BATCH_SIZE = 17
stage2_num_epochs = 20
base_lr = 0.004
VAL_INTERVAL = 1

default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=2, save_best='coco/bbox_mAP'),
    # auto coco/bbox_mAP_50 coco/bbox_mAP_75 coco/bbox_mAP_s
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=MAX_EPOCHS,
    val_interval=VAL_INTERVAL,
    dynamic_intervals=[(MAX_EPOCHS - stage2_num_epochs, 1)])

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=150,
        end=300,
        T_max=150,
        by_epoch=True,
        convert_to_iter_based=True)
]

val_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'validation/label.json',
    ann_file=data_root + 'coco_validation/filtered_instances.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
auto_scale_lr = dict(enable=False, base_batch_size=16)

# DataLoader
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True,with_attribute = True),
    # dict(
    #     type='CachedMosaic',
    #     img_scale=(640, 640),
    #     pad_val=114.0,
    #     max_cached_images=20,
    #     random_pop=False),
    dict(
        type='Resize',
        scale=(1280, 1280)),
    # dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    # dict(
    #     type='CachedMixUp',
    #     img_scale=(640, 640),
    #     ratio_range=(1.0, 1.0),
    #     max_cached_images=10,
    #     random_pop=False,
    #     pad_val=(114, 114, 114),
    #     prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='coco_train/filtered_instances.json',
        data_prefix=dict(img='coco_train/data/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None),
    pin_memory=True)
val_dataloader = dict(
    batch_size=VAL_BATCH_SIZE,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='coco_validation/filtered_instances.json',            
        data_prefix=dict(img='coco_validation/data/'),   
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None))
test_dataloader = val_dataloader


model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        # init_cfg=dict(
        #     # type='Pretrained',
        #     prefix='backbone.',
        #     # checkpoint=backbone_pretrain

        # )
        ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=in_channels,
        out_channels=neck_out_channels,
        num_csp_blocks=num_csp_blocks,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHeadSORA',
        num_classes=6,
        in_channels=neck_out_channels,
        stacked_convs=2,
        feat_channels=neck_out_channels,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=exp_on_reg,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))
