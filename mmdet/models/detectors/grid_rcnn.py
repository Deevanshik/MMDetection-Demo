import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..registry import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class GridRCNN(TwoStageDetector):
    """Grid R-CNN.

    This detector is the implementation of:
    - Grid R-CNN (https://arxiv.org/abs/1811.12030)
    - Grid R-CNN Plus: Faster and Better (https://arxiv.org/abs/1906.05688)
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        assert grid_head is not None
        super(GridRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
