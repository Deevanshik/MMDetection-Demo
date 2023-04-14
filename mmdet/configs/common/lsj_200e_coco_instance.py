if '_base_':
    from .lsj_100e_coco_instance import *
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

# 8x25=200e
train_dataloader.merge(dict(dataset=dict(times=8)))

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.067, by_epoch=False, begin=0, end=1000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=25,
        by_epoch=True,
        milestones=[22, 24],
        gamma=0.1)
]
