from abc import ABCMeta, abstractmethod

import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32

from ..builder import build_loss
from ..utils import upsample_like


class BaseSemanticHead(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 num_classes,
                 num_feats=-1,
                 loss_semantic=dict(
                     type='CrossEntropyLoss', ignore_index=-1,
                     loss_weight=1.0)):
        super(BaseSemanticHead, self).__init__()
        self.loss_semantic = build_loss(loss_semantic)
        self.num_classes = num_classes
        self.num_feats = num_feats
        self.eps = 1e-6

    @force_fp32(apply_to=('logits', ))
    def loss(self, logits, gt_semantic_seg):
        if logits.shape[-2:] != gt_semantic_seg.shape[-2:]:
            logits = upsample_like(logits, gt_semantic_seg)
        logits = logits.permute((0, 2, 3, 1))
        # hard code here, minus one
        gt_semantic_seg = gt_semantic_seg - 1

        loss_semantic = self.loss_semantic(
            logits.reshape(-1, self.num_classes),  # => [NxHxW, C]
            gt_semantic_seg.reshape(-1).long())
        return dict(loss_semantic=loss_semantic)

    @auto_fp16()
    @abstractmethod
    def forward(self, x):
        """Placeholder of forward function.

        Returns:
            dict[str, Tensor]: A dictionary, including features
                and predicted scores. Required keys: 'fcn_scores'
                and 'fcn_feats'.
        """
        pass

    def forward_train(self, x, gt_semantic_seg):
        fcn_output = self.forward(x[:self.num_feats])
        logits = fcn_output['fcn_score']
        return self.loss(logits, gt_semantic_seg)

    def simple_test(self, x, img_metas, rescale=False):
        fcn_output = self.forward(x[:self.num_feats])
        logits = fcn_output['fcn_score']
        logits = F.interpolate(
            logits,
            size=img_metas[0]['pad_shape'][:2],
            mode='bilinear',
            align_corners=False)

        if rescale:
            h, w, _ = img_metas[0]['img_shape']
            logits = logits[:, :, :h, :w]

            h, w, _ = img_metas[0]['ori_shape']
            logits = F.interpolate(
                logits, size=(h, w), mode='bilinear', align_corners=False)
        return logits
