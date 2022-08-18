_base_ = '../dcn/cascade-mask-rcnn_x101-32x4d-fpn-dconv-c3-c5_1x_coco.py'
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True), norm_eval=False))
