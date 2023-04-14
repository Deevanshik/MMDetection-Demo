if '_base_':
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.models.detectors.lad import LAD
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.models.backbones.resnet import ResNet, ResNet
from mmdet.models.necks.fpn import FPN, FPN
from mmdet.models.dense_heads.lad_head import LADHead, LADHead
from mmdet.models.task_modules.prior_generators.anchor_generator import AnchorGenerator, AnchorGenerator
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder, DeltaXYWHBBoxCoder
from mmdet.models.losses.focal_loss import FocalLoss, FocalLoss
from mmdet.models.losses.iou_loss import GIoULoss, GIoULoss
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss, CrossEntropyLoss
from mmdet.models.task_modules.assigners.max_iou_assigner import MaxIoUAssigner
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth'  # noqa

model = dict(
    type=LAD,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    # student
    backbone=dict(
        type=ResNet,
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        type=FPN,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type=LADHead,
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type=AnchorGenerator,
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type=DeltaXYWHBBoxCoder,
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type=GIoULoss, loss_weight=1.3),
        loss_centerness=dict(
            type=CrossEntropyLoss, use_sigmoid=True, loss_weight=0.5)),
    # teacher
    teacher_ckpt=teacher_ckpt,
    teacher_backbone=dict(
        type=ResNet,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    teacher_neck=dict(
        type=FPN,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    teacher_bbox_head=dict(
        type=LADHead,
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type=AnchorGenerator,
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type=DeltaXYWHBBoxCoder,
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type=FocalLoss,
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type=GIoULoss, loss_weight=1.3),
        loss_centerness=dict(
            type=CrossEntropyLoss, use_sigmoid=True, loss_weight=0.5)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type=MaxIoUAssigner,
            pos_iou_thr=0.1,
            neg_iou_thr=0.1,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        score_voting=True,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
train_dataloader.merge(dict(batch_size=8, num_workers=4))
optim_wrapper.merge(dict(type=AmpOptimWrapper, optimizer=dict(lr=0.01)))
