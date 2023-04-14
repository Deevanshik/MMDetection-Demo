if '_base_':
    from ..common.ms_poly_3x_coco_instance import *
    from .._base_.models.mask_rcnn_r50_fpn import *
from mmdet.models.backbones.regnet import RegNet
from mmdet.models.necks.fpn import FPN
from torch.optim.sgd import SGD

model.merge(
    dict(
        backbone=dict(
            _delete_=True,
            type=RegNet,
            arch='regnetx_1.6gf',
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://regnetx_1.6gf')),
        neck=dict(
            type=FPN,
            in_channels=[72, 168, 408, 912],
            out_channels=256,
            num_outs=5)))

optim_wrapper.merge(
    dict(
        optimizer=dict(type=SGD, lr=0.02, momentum=0.9, weight_decay=0.00005),
        clip_grad=dict(max_norm=35, norm_type=2)))
