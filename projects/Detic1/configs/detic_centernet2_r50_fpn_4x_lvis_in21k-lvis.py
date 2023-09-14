_base_ = 'mmdet::_base_/default_runtime.py'
dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis/'
custom_imports = dict(
    imports=['projects.Detic1.detic'], allow_failed_imports=False)

num_classes = 1203
lvis_v1_train_cat_info = 'data/metadata/lvis_v1_train_cat_info.json'
curl = 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth'
image_size_det = (640, 640)
image_size_cls = (320, 320)
# batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]

cls_layer = dict(
    type='ZeroShotClassifier',
    zs_weight_path='data/metadata/lvis_v1_clip_a+cname.npy',
    zs_weight_dim=512,
    use_bias=0.0,
    norm_weight=True,
    norm_temperature=50.0)
reg_layer = [
    dict(type='Linear', in_features=1024, out_features=1024),
    dict(type='ReLU', inplace=True),
    dict(type='Linear', in_features=1024, out_features=4)
]

model = dict(
    type='Detic',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    # batch_augments=batch_augments),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=curl,
        )),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    rpn_head=dict(
        type='CenterNetRPNHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        conv_bias=True,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_cls=dict(
            type='HeatmapFocalLoss',
            alpha=0.25,
            beta=4.0,
            gamma=2.0,
            pos_weight=0.5,
            neg_weight=0.5,
            loss_weight=1.0,
            ignore_high_fp=0.85,
        ),
        loss_bbox=dict(type='GIoULoss', eps=1e-6, loss_weight=1.0),
    ),
    roi_head=dict(
        type='DeticRoIHead',
        num_stages=3,
        stage_loss_weights=[1.0, 1.0, 1.0],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=0,
                use_torchvision=True),
            out_channels=256,
            featmap_strides=[8, 16, 32],
            # approximately equal to
            # canonical_box_size=224, canonical_level=4 in D2
            finest_scale=112),
        bbox_head=[
            dict(
                type='DeticBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                cls_predictor_cfg=cls_layer,
                reg_predictor_cfg=reg_layer,
                use_fed_loss=True,
                cat_freq_path=lvis_v1_train_cat_info,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=0.1,
                               loss_weight=1.0)),
            dict(
                type='DeticBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                cls_predictor_cfg=cls_layer,
                reg_predictor_cfg=reg_layer,
                use_fed_loss=True,
                cat_freq_path=lvis_v1_train_cat_info,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=0.1,
                               loss_weight=1.0)),
            dict(
                type='DeticBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                cls_predictor_cfg=cls_layer,
                reg_predictor_cfg=reg_layer,
                use_fed_loss=True,
                cat_freq_path=lvis_v1_train_cat_info,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=0.1, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32],
            # approximately equal to
            # canonical_box_size=224, canonical_level=4 in D2
            finest_scale=112),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            class_agnostic=True,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            score_thr=0.0001,
            nms_pre=4000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.9),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.8,
                    neg_iou_thr=0.8,
                    min_pos_iou=0.8,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            score_thr=0.0001,
            nms_pre=1000,
            max_per_img=256,
            nms=dict(type='nms', iou_threshold=0.9),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.02,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5)))

# backend = 'pillow'
backend_args = None

train_pipeline_det = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=image_size_det,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size_det,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_pipeline_cls = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=False, with_label=True),
    dict(
        type='RandomResize',
        scale=image_size_cls,
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size_cls,
        recompute_bbox=False,
        bbox_clip_border=False,
        allow_negative_crop=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args,
        imdecode_backend=backend_args),
    dict(
        type='Resize',
        scale=(1333, 800),
        keep_ratio=True,
        backend=backend_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

dataset_det = dict(
    type='ClassBalancedDataset',
    oversample_thr=1e-3,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/lvis_v1_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline_det,
        backend_args=backend_args))

dataset_cls = dict(
    type='IMAGENETLVISV1Dataset',
    data_root='data/imagenet',
    ann_file='annotations/imagenet_lvis_image_info.json',
    data_prefix=dict(img='ImageNet-LVIS/'),
    pipeline=train_pipeline_cls,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=[8, 32],
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='MultiDataSampler', dataset_ratio=[1, 4]),
    batch_sampler=dict(type='MDAspectRatioBatchSampler', num_datasets=2),
    dataset=dict(type='MultiDataDataset', datasets=[dataset_det, dataset_cls]))

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/lvis_v1_val.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline,
        return_classes=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='LVISMetric',
    ann_file=data_root + 'annotations/lvis_v1_val.json',
    metric=['bbox', 'segm'])
test_evaluator = val_evaluator

# training schedule for 90k
max_iter = 90000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iter, val_interval=90000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0.),
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        by_epoch=False,
        T_max=90000,
    )
]

# only keep latest 5 checkpoints
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=30000, max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=50))

load_from = './first_stage/boxsup_centernet2_r50_fpn_4x_lvis.pth'

find_unused_parameters = True
