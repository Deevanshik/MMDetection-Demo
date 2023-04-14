if '_base_':
    from .._base_.models.retinanet_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from torch.optim.sgd import SGD

# model
model.merge(
    dict(
        backbone=dict(
            depth=18,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet18')),
        neck=dict(in_channels=[64, 128, 256, 512])))
optim_wrapper.merge(
    dict(optimizer=dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0001)))

# TODO: support auto scaling lr
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
# auto_scale_lr = dict(base_batch_size=16)
