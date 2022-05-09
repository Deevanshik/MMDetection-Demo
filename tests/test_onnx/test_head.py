# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from functools import partial

import mmcv
import numpy as np
import pytest
import torch
from mmcv.cnn import Scale

from mmdet import digit_version
from mmdet.models import build_detector
from mmdet.models.dense_heads import (FCOSHead, FSAFHead, RetinaHead, SSDHead,
                                      YOLOV3Head)
from .utils import ort_validate

data_path = osp.join(osp.dirname(__file__), 'data')

if digit_version(torch.__version__) <= digit_version('1.5.0'):
    pytest.skip(
        'ort backend does not support version below 1.5.0',
        allow_module_level=True)


def test_cascade_onnx_export():

    config_path = './configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
    cfg = mmcv.Config.fromfile(config_path)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    with torch.no_grad():
        model.forward = partial(model.forward, img_metas=[[dict()]])

        dynamic_axes = {
            'input_img': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'dets': {
                0: 'batch',
                1: 'num_dets',
            },
            'labels': {
                0: 'batch',
                1: 'num_dets',
            },
        }
        torch.onnx.export(
            model, [torch.rand(1, 3, 400, 500)],
            'tmp.onnx',
            output_names=['dets', 'labels'],
            input_names=['input_img'],
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11,
            dynamic_axes=dynamic_axes)


def test_cascade_onnx_export_normalize_in_graph():

    config_path = './configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
    cfg = mmcv.Config.fromfile(config_path)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    with torch.no_grad():
        model.forward = partial(
            model.forward,
            img_metas=[[{
                'img_norm_cfg': {
                    'mean':
                    np.array(cfg['img_norm_cfg']['mean'], dtype=np.float32),
                    'std':
                    np.array(cfg['img_norm_cfg']['std'], dtype=np.float32),
                }
            }]])

        dynamic_axes = {
            'input_img': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'dets': {
                0: 'batch',
                1: 'num_dets',
            },
            'labels': {
                0: 'batch',
                1: 'num_dets',
            },
        }
        torch.onnx.export(
            model, [torch.rand(1, 3, 400, 500)],
            'tmp.onnx',
            output_names=['dets', 'labels'],
            input_names=['input_img'],
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11,
            dynamic_axes=dynamic_axes)


def test_faster_onnx_export():

    config_path = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    cfg = mmcv.Config.fromfile(config_path)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    with torch.no_grad():
        model.forward = partial(model.forward, img_metas=[[dict()]])

        dynamic_axes = {
            'input_img': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'dets': {
                0: 'batch',
                1: 'num_dets',
            },
            'labels': {
                0: 'batch',
                1: 'num_dets',
            },
        }
        torch.onnx.export(
            model, [torch.rand(1, 3, 400, 500)],
            'tmp.onnx',
            output_names=['dets', 'labels'],
            input_names=['input_img'],
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11,
            dynamic_axes=dynamic_axes)


def retinanet_config():
    """RetinanNet Head Config."""
    head_cfg = dict(
        stacked_convs=6,
        feat_channels=2,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]))

    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    model = RetinaHead(
        num_classes=4, in_channels=1, test_cfg=test_cfg, **head_cfg)
    model.requires_grad_(False)

    return model


def test_retina_head_forward_single():
    """Test RetinaNet Head single forward in torch and onnxruntime env."""
    retina_model = retinanet_config()

    feat = torch.rand(1, retina_model.in_channels, 32, 32)
    # validate the result between the torch and ort
    ort_validate(retina_model.forward_single, feat)


def test_retina_head_forward():
    """Test RetinaNet Head forward in torch and onnxruntime env."""
    retina_model = retinanet_config()
    s = 128
    # RetinaNet head expects a multiple levels of features per image
    feats = [
        torch.rand(1, retina_model.in_channels, s // (2**(i + 2)),
                   s // (2**(i + 2)))  # [32, 16, 8, 4, 2]
        for i in range(len(retina_model.prior_generator.strides))
    ]
    ort_validate(retina_model.forward, feats)


def test_retinanet_head_onnx_export():
    """Test RetinaNet Head _get_bboxes() in torch and onnxruntime env."""
    retina_model = retinanet_config()
    s = 128
    img_metas = [{
        'img_shape_for_onnx': torch.Tensor([s, s]),
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 2)
    }]

    # The data of retina_head_get_bboxes.pkl contains two parts:
    # cls_score(list(Tensor)) and bboxes(list(Tensor)),
    # where each torch.Tensor is generated by torch.rand().
    # the cls_score's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2).
    # the bboxes's size: (1, 36, 32, 32), (1, 36, 16, 16),
    # (1, 36, 8, 8), (1, 36, 4, 4), (1, 36, 2, 2)
    retina_head_data = 'retina_head_get_bboxes.pkl'
    feats = mmcv.load(osp.join(data_path, retina_head_data))
    cls_score = feats[:5]
    bboxes = feats[5:]

    retina_model.onnx_export = partial(
        retina_model.onnx_export, img_metas=img_metas, with_nms=False)
    ort_validate(retina_model.onnx_export, (cls_score, bboxes))


def yolo_config():
    """YoloV3 Head Config."""
    head_cfg = dict(
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'))

    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            conf_thr=0.005,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=100))

    model = YOLOV3Head(
        num_classes=4,
        in_channels=[1, 1, 1],
        out_channels=[16, 8, 4],
        test_cfg=test_cfg,
        **head_cfg)
    model.requires_grad_(False)
    # yolov3 need eval()
    model.cpu().eval()
    return model


def test_yolov3_head_forward():
    """Test Yolov3 head forward() in torch and ort env."""
    yolo_model = yolo_config()

    # Yolov3 head expects a multiple levels of features per image
    feats = [
        torch.rand(1, 1, 64 // (2**(i + 2)), 64 // (2**(i + 2)))
        for i in range(len(yolo_model.in_channels))
    ]
    ort_validate(yolo_model.forward, feats)


def test_yolov3_head_onnx_export():
    """Test yolov3 head get_bboxes() in torch and ort env."""
    yolo_model = yolo_config()
    s = 128
    img_metas = [{
        'img_shape_for_onnx': torch.Tensor([s, s]),
        'img_shape': (s, s, 3),
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3)
    }]

    # The data of yolov3_head_get_bboxes.pkl contains
    # a list of torch.Tensor, where each torch.Tensor
    # is generated by torch.rand and each tensor size is:
    # (1, 27, 32, 32), (1, 27, 16, 16), (1, 27, 8, 8).
    yolo_head_data = 'yolov3_head_get_bboxes.pkl'
    pred_maps = mmcv.load(osp.join(data_path, yolo_head_data))

    yolo_model.onnx_export = partial(
        yolo_model.onnx_export, img_metas=img_metas, with_nms=False)
    ort_validate(yolo_model.onnx_export, pred_maps)


def fcos_config():
    """FCOS Head Config."""
    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    model = FCOSHead(num_classes=4, in_channels=1, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


def test_fcos_head_forward_single():
    """Test fcos forward single in torch and ort env."""
    fcos_model = fcos_config()

    feat = torch.rand(1, fcos_model.in_channels, 32, 32)
    fcos_model.forward_single = partial(
        fcos_model.forward_single,
        scale=Scale(1.0).requires_grad_(False),
        stride=(4, ))
    ort_validate(fcos_model.forward_single, feat)


def test_fcos_head_forward():
    """Test fcos forward in mutil-level feature map."""
    fcos_model = fcos_config()
    s = 128
    feats = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    ort_validate(fcos_model.forward, feats)


def test_fcos_head_onnx_export():
    """Test fcos head get_bboxes() in ort."""
    fcos_model = fcos_config()
    s = 128
    img_metas = [{
        'img_shape_for_onnx': torch.Tensor([s, s]),
        'img_shape': (s, s, 3),
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3)
    }]

    cls_scores = [
        torch.rand(1, fcos_model.num_classes, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    bboxes = [
        torch.rand(1, 4, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]
    centerness = [
        torch.rand(1, 1, s // feat_size, s // feat_size)
        for feat_size in [4, 8, 16, 32, 64]
    ]

    fcos_model.onnx_export = partial(
        fcos_model.onnx_export, img_metas=img_metas, with_nms=False)
    ort_validate(fcos_model.onnx_export, (cls_scores, bboxes, centerness))


def fsaf_config():
    """FSAF Head Config."""
    cfg = dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=1,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]))

    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    model = FSAFHead(num_classes=4, in_channels=1, test_cfg=test_cfg, **cfg)
    model.requires_grad_(False)
    return model


def test_fsaf_head_forward_single():
    """Test RetinaNet Head forward_single() in torch and onnxruntime env."""
    fsaf_model = fsaf_config()

    feat = torch.rand(1, fsaf_model.in_channels, 32, 32)
    ort_validate(fsaf_model.forward_single, feat)


def test_fsaf_head_forward():
    """Test RetinaNet Head forward in torch and onnxruntime env."""
    fsaf_model = fsaf_config()
    s = 128
    feats = [
        torch.rand(1, fsaf_model.in_channels, s // (2**(i + 2)),
                   s // (2**(i + 2)))
        for i in range(len(fsaf_model.anchor_generator.strides))
    ]
    ort_validate(fsaf_model.forward, feats)


def test_fsaf_head_onnx_export():
    """Test RetinaNet Head get_bboxes in torch and onnxruntime env."""
    fsaf_model = fsaf_config()
    s = 256
    img_metas = [{
        'img_shape_for_onnx': torch.Tensor([s, s]),
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 2)
    }]

    # The data of fsaf_head_get_bboxes.pkl contains two parts:
    # cls_score(list(Tensor)) and bboxes(list(Tensor)),
    # where each torch.Tensor is generated by torch.rand().
    # the cls_score's size: (1, 4, 64, 64), (1, 4, 32, 32),
    # (1, 4, 16, 16), (1, 4, 8, 8), (1, 4, 4, 4).
    # the bboxes's size: (1, 4, 64, 64), (1, 4, 32, 32),
    # (1, 4, 16, 16), (1, 4, 8, 8), (1, 4, 4, 4).
    fsaf_head_data = 'fsaf_head_get_bboxes.pkl'
    feats = mmcv.load(osp.join(data_path, fsaf_head_data))
    cls_score = feats[:5]
    bboxes = feats[5:]

    fsaf_model.onnx_export = partial(
        fsaf_model.onnx_export, img_metas=img_metas, with_nms=False)
    ort_validate(fsaf_model.onnx_export, (cls_score, bboxes))


def ssd_config():
    """SSD Head Config."""
    cfg = dict(
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=300,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]))

    test_cfg = mmcv.Config(
        dict(
            deploy_nms_pre=0,
            nms=dict(type='nms', iou_threshold=0.45),
            min_bbox_size=0,
            score_thr=0.02,
            max_per_img=200))

    model = SSDHead(
        num_classes=4,
        in_channels=(4, 8, 4, 2, 2, 2),
        test_cfg=test_cfg,
        **cfg)

    model.requires_grad_(False)
    return model


def test_ssd_head_forward():
    """Test SSD Head forward in torch and onnxruntime env."""
    ssd_model = ssd_config()

    featmap_size = [38, 19, 10, 6, 5, 3, 1]

    feats = [
        torch.rand(1, ssd_model.in_channels[i], featmap_size[i],
                   featmap_size[i]) for i in range(len(ssd_model.in_channels))
    ]
    ort_validate(ssd_model.forward, feats)


def test_ssd_head_onnx_export():
    """Test SSD Head get_bboxes in torch and onnxruntime env."""
    ssd_model = ssd_config()
    s = 300
    img_metas = [{
        'img_shape_for_onnx': torch.Tensor([s, s]),
        'scale_factor': np.ones(4),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 2)
    }]

    # The data of ssd_head_get_bboxes.pkl contains two parts:
    # cls_score(list(Tensor)) and bboxes(list(Tensor)),
    # where each torch.Tensor is generated by torch.rand().
    # the cls_score's size: (1, 20, 38, 38), (1, 30, 19, 19),
    # (1, 30, 10, 10), (1, 30, 5, 5), (1, 20, 3, 3), (1, 20, 1, 1).
    # the bboxes's size: (1, 16, 38, 38), (1, 24, 19, 19),
    # (1, 24, 10, 10), (1, 24, 5, 5), (1, 16, 3, 3), (1, 16, 1, 1).
    ssd_head_data = 'ssd_head_get_bboxes.pkl'
    feats = mmcv.load(osp.join(data_path, ssd_head_data))
    cls_score = feats[:6]
    bboxes = feats[6:]

    ssd_model.onnx_export = partial(
        ssd_model.onnx_export, img_metas=img_metas, with_nms=False)
    ort_validate(ssd_model.onnx_export, (cls_score, bboxes))
