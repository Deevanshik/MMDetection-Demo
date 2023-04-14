if '_base_':
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.models.detectors.kd_one_stage import KnowledgeDistillationSingleStageDetector
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.necks.fpn import FPN
from mmdet.models.dense_heads.ld_head import LDHead
from mmdet.models.task_modules.prior_generators.anchor_generator import AnchorGenerator
from mmdet.models.losses.gfocal_loss import QualityFocalLoss, DistributionFocalLoss
from mmdet.models.losses.kd_loss import KnowledgeDistillationKLDivLoss
from mmdet.models.losses.iou_loss import GIoULoss
from mmdet.models.task_modules.assigners.atss_assigner import ATSSAssigner
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from torch.optim.sgd import SGD

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'  # noqa
model = dict(
    type=KnowledgeDistillationSingleStageDetector,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    teacher_config='configs/gfl/gfl_r101_fpn_ms-2x_coco.py',
    teacher_ckpt=teacher_ckpt,
    backbone=dict(
        type=ResNet,
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type=FPN,
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type=LDHead,
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
        loss_cls=dict(
            type=QualityFocalLoss, use_sigmoid=True, beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type=DistributionFocalLoss, loss_weight=0.25),
        loss_ld=dict(
            type=KnowledgeDistillationKLDivLoss, loss_weight=0.25, T=10),
        reg_max=16,
        loss_bbox=dict(type=GIoULoss, loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type=ATSSAssigner, topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

optim_wrapper.merge(
    dict(
        type=OptimWrapper,
        optimizer=dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0001)))
