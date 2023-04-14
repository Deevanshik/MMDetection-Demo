if '_base_':
    from .._base_.models.retinanet_r50_fpn import *
    from .._base_.datasets.objects365v1_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from torch.optim.sgd import SGD
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

model.merge(dict(bbox_head=dict(num_classes=365)))

# Using 8 GPUS while training
optim_wrapper.merge(
    dict(
        type=OptimWrapper,
        optimizer=dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0001),
        clip_grad=dict(max_norm=35, norm_type=2)))

# learning rate
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=10000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=16))
