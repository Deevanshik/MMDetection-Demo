if '_base_':
    from .._base_.models.mask_rcnn_r50_fpn import *
    from .._base_.datasets.coco_instance import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import RandomFlip, RandomCrop
from mmcv.transforms.wrappers import RandomChoice
from mmcv.transforms.processing import RandomChoiceResize, RandomChoiceResize, RandomChoiceResize
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper
from torch.optim.adamw import AdamW

# TODO: delete custom_imports after mmcls supports auto import
# please install mmcls>=1.0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

model.merge(
    dict(
        backbone=dict(
            _delete_=True,
            type='mmcls.ConvNeXt',
            arch='tiny',
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            init_cfg=dict(
                type='Pretrained',
                checkpoint=checkpoint_file,
                prefix='backbone.')),
        neck=dict(in_channels=[96, 192, 384, 768])))

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=RandomChoice,
        transforms=[[
            dict(
                type=RandomChoiceResize,
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type=RandomChoiceResize,
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type=RandomCrop,
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type=RandomChoiceResize,
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(type=PackDetInputs)
]
train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))

max_epochs = 36
train_cfg.merge(dict(max_epochs=max_epochs))

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper.merge(
    dict(
        type=AmpOptimWrapper,
        constructor='LearningRateDecayOptimizerConstructor',
        paramwise_cfg={
            'decay_rate': 0.95,
            'decay_type': 'layer_wise',
            'num_layers': 6
        },
        optimizer=dict(
            _delete_=True,
            type=AdamW,
            lr=0.0001,
            betas=(0.9, 0.999),
            weight_decay=0.05,
        )))
