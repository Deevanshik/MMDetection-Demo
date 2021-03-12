import os.path as osp
import pickle
from functools import partial

import mmcv
import numpy as np
import pytest
import torch

from mmdet import digit_version
from mmdet.models.dense_heads import RetinaHead, YOLOV3Head
from .utils import (WrapFunction, convert_result_list, ort_validate,
                    verify_model)

data_path = osp.join(osp.dirname(__file__), 'data')

if digit_version(torch.__version__) <= digit_version('1.5.0'):
    pytest.skip(
        'ort backend does not support version below 1.5.0',
        allow_module_level=True)


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
            deploy_nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    model = RetinaHead(
        num_classes=4, in_channels=1, test_cfg=test_cfg, **head_cfg)
    model.requires_grad_(False)
    model.eval()

    return model


def test_retina_head_forward_single():
    """Test RetinaNet Head single forward in torch and onnxruntime env."""
    retina_model = retinanet_config()

    feat = torch.rand(1, retina_model.in_channels, 32, 32)
    wrap_model = WrapFunction(retina_model.forward_single)
    wrap_model.cpu().eval()
    ort_validate(wrap_model, feat)


def test_retina_head_forward():
    """Test RetinaNet Head forward in torch and onnxruntime env."""

    retina_model = retinanet_config()
    s = 128

    # RetinaNet head expects a multiple levels of features per image
    feats = [
        torch.rand(1, retina_model.in_channels, s // (2**(i + 2)),
                   s // (2**(i + 2)))
        for i in range(len(retina_model.anchor_generator.strides))
    ]

    wrap_model = WrapFunction(retina_model.forward)
    wrap_model.cpu().eval()
    ort_validate(wrap_model, feats)


def test_retinanet_head_get_bboxes():
    """Test RetinaNet Head _get_bboxes() in torch and onnxruntime env."""

    retina_model = retinanet_config()
    s = 128
    img_metas = [{
        'img_shape_for_onnx': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 2)
    }]

    # The data of retina_head_get_bboxes.pkl contains two parts:
    # cls_score(list(Tensor)) and bboxes(list(Tensor)), where each
    # torch.Tensor is generated by torch.rand().
    # the cls_score's size: (1, 36, 32, 32), (1, 36, 16, 16), (1, 36, 8, 8),
    # (1, 36, 4, 4), (1, 36, 2, 2).
    # the bboxes's size: (1, 36, 32, 32), (1, 36, 16, 16), (1, 36, 8, 8),
    # (1, 36, 4, 4), (1, 36, 2, 2)
    retina_head_data = 'retina_head_get_bboxes.pkl'
    with open(osp.join(data_path, retina_head_data), 'rb') as f:
        feats = pickle.load(f)
    cls_score = feats[:5]
    bboxes = feats[5:]

    retina_model.get_bboxes = partial(
        retina_model.get_bboxes, img_metas=img_metas)
    wrap_model = WrapFunction(retina_model.get_bboxes)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model, (cls_score, bboxes),
            'tmp.onnx',
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(cls_score + bboxes)

    torch_outputs = wrap_model.forward(cls_score, bboxes)
    torch_outputs = convert_result_list(torch_outputs)
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)


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
            deploy_nms_pre=1000,
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
    model.eval()

    return model


def test_yolov3_head_forward():
    """Test Yolov3 head forward() in torch and ort env."""

    yolo_model = yolo_config()

    # Yolov3 head expects a multiple levels of features per image
    feats = [
        torch.rand(1, 1, 64 // (2**(i + 2)), 64 // (2**(i + 2)))
        for i in range(len(yolo_model.in_channels))
    ]

    wrap_model = WrapFunction(yolo_model.forward)
    wrap_model.cpu().eval()
    ort_validate(wrap_model, feats)


def test_yolov3_head_get_bboxes():
    """Test yolov3 head get_bboxes() in torch and ort env."""

    yolo_model = yolo_config()

    s = 128
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    # The data of yolov3_head_get_bboxes.pkl contains a list of
    # torch.Tensor, where each torch.Tensor is generated by
    # torch.rand and each tensor size is:
    # (1, 27, 32, 32), (1, 27, 16, 16), (1, 27, 8, 8).
    yolo_head_data = 'yolov3_head_get_bboxes.pkl'
    with open(osp.join(data_path, yolo_head_data), 'rb') as f:
        pred_maps = pickle.load(f)

    yolo_model.get_bboxes = partial(yolo_model.get_bboxes, img_metas=img_metas)
    wrap_model = WrapFunction(yolo_model.get_bboxes)
    wrap_model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            wrap_model,
            pred_maps,
            'tmp.onnx',
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

    onnx_outputs = verify_model(pred_maps)

    torch_outputs = convert_result_list(wrap_model.forward(pred_maps))
    torch_outputs = [
        torch_output.detach().numpy() for torch_output in torch_outputs
    ]

    # match torch_outputs and onnx_outputs
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            torch_outputs[i], onnx_outputs[i], rtol=1e-03, atol=1e-05)
