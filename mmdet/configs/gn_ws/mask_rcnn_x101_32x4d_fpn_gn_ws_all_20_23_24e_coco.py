if '_base_':
    from .mask_rcnn_x101_32x4d_fpn_gn_ws_all_2x_coco import *
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
# learning policy
max_epochs = 24
train_cfg.merge(dict(max_epochs=max_epochs))

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20, 23],
        gamma=0.1)
]
