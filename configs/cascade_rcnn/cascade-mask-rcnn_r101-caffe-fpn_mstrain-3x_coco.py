_base_ = './cascade-mask-rcnn_r50-caffe-fpn_mstrain-3x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')))
