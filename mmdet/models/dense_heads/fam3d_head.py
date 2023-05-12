# Copyright (c) OpenMMLab. All rights reserved.
from math import pi
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import DeformConv2d
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules import anchor_inside_flags, build_assigner
from mmdet.models.utils import images_to_levels, multi_apply, unmap
from mmdet.registry import MODELS
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList)
from ..utils import filter_scores_and_topk
from . import ATSSHead


@MODELS.register_module()
class FAM3DHead(ATSSHead):
    """
    Args:
        num_dcn (int): Number of deformable convolution in the head.
            Default: 0.
        anchor_type (str): If set to `anchor_free`, the head will use centers
            to regress bboxes. If set to `anchor_based`, the head will
            regress bboxes based on anchors. Default: `anchor_free`.

    Example:
        >>> self = FAM3DHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 num_dcn: int = 0,
                 anchor_type: str = 'anchor_free',
                 use_atan: bool = False,
                 offset_channel_shrink: int = 4,
                 **kwargs) -> None:
        assert anchor_type in ['anchor_free', 'anchor_based']
        self.num_dcn = num_dcn
        self.anchor_type = anchor_type
        self.use_atan = use_atan
        self.offset_channel_shrink = offset_channel_shrink
        self.epoch = 0  # which would be update in SetEpochInfoHook!
        super().__init__(num_classes, in_channels, **kwargs)

        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.alignment_assigner = build_assigner(self.train_cfg.assigner)
            self.alpha = self.train_cfg.alpha
            self.beta = self.train_cfg.beta

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.tood_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.tood_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)
        self.pyramid_mapping_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs * 2,
                      self.feat_channels // self.offset_channel_shrink, 1),
            nn.ReLU(inplace=True))
        self.reg_offset_module = nn.Conv2d(
            self.feat_channels // self.offset_channel_shrink * 3,
            4 * 2,
            3,
            padding=1)
        self.pyramid_offset = nn.Conv2d(
            self.feat_channels // self.offset_channel_shrink * 3,
            4,
            3,
            padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.pyramid_mapping_module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.tood_cls, std=0.01, bias=bias_cls)
        normal_init(self.tood_reg, std=0.01)
        normal_init(self.reg_offset_module, std=0.001)
        normal_init(self.pyramid_offset, std=0.001)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Decoded box for all scale levels,
                    each is a 4D-tensor, the channels number is
                    num_anchors * 4. In [tl_x, tl_y, br_x, br_y] format.
        """
        cls_scores = []
        bbox_preds = []
        offset_feats = []
        for idx, (x, scale, stride) in enumerate(
                zip(x, self.scales, self.prior_generator.strides)):
            b, c, h, w = x.shape
            anchor = self.prior_generator.single_level_grid_priors(
                (h, w), idx, device=x.device)
            anchor = torch.cat([anchor for _ in range(b)])

            cls_feat = x
            reg_feat = x
            inter_feat = []
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
                inter_feat.append(cls_feat)

            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
                inter_feat.append(reg_feat)
            inter_feat = torch.cat(inter_feat, 1)

            cls_score = self.tood_cls(cls_feat).sigmoid()

            # reg prediciton and alignment
            if self.anchor_type == 'anchor_free':
                reg_dist = scale(self.tood_reg(reg_feat).exp()).float()
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
                reg_bbox = distance2bbox(
                    self.anchor_center(anchor) / stride[0],
                    reg_dist).reshape(b, h, w, 4).permute(0, 3, 1,
                                                          2)  # (b, c, h, w)
            elif self.anchor_type == 'anchor_based':
                reg_dist = scale(self.tood_reg(reg_feat)).float()
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
                reg_bbox = self.bbox_coder.decode(anchor, reg_dist).reshape(
                    b, h, w, 4).permute(0, 3, 1, 2) / stride[0]
            else:
                raise NotImplementedError(
                    f'Unknow anchor type: {self.anchor_type}.'
                    f'Please use either `anchor_free` or `anchor_based`.')

            cls_scores.append(cls_score)
            bbox_preds.append(reg_bbox)
            offset_feats.append(self.pyramid_mapping_module(inter_feat))

        bbox_preds_new = []
        for idx in range(len(x)):
            b, c, h, w = offset_feats[idx].shape
            if idx > 0:
                lower = F.interpolate(
                    offset_feats[idx - 1],
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False)
            else:
                lower = offset_feats[idx]

            if idx < len(x) - 1:
                upper = F.interpolate(
                    offset_feats[idx + 1],
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False)
            else:
                upper = offset_feats[idx]

            offset_feat = torch.cat([lower, offset_feats[idx], upper], dim=1)
            weight = self.pyramid_offset(offset_feat)

            if self.use_atan:
                weight = weight.atan() / (pi / 2)
                if idx == 0:
                    weight = weight.clamp(min=0)
                if idx == len(x) - 1:
                    weight = weight.clamp(max=0)
            else:
                if idx == 0:
                    weight = weight.clamp(min=-1)
                else:
                    weight = weight.clamp(max=0)
                if idx == len(x) - 1:
                    weight = weight.clamp(max=0)
                else:
                    weight = weight.clamp(max=1)
            weight_curr = 1 - weight.abs()
            weight_top = weight.clamp(min=0)
            weight_bottom = (-weight).clamp(min=0)
            bbox_pred = bbox_preds[idx] * weight_curr
            if idx > 0:
                bbox_pred = bbox_pred + weight_bottom * F.interpolate(
                    bbox_preds[idx - 1],
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False) * self.prior_generator.strides[
                        idx - 1][0] / self.prior_generator.strides[idx][0]
            if idx < len(x) - 1:
                bbox_pred = bbox_pred + weight_top * F.interpolate(
                    bbox_preds[idx + 1],
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False) * self.prior_generator.strides[
                        idx + 1][0] / self.prior_generator.strides[idx][0]
            reg_offset = self.reg_offset_module(offset_feat)
            bbox_pred = self.deform_sampling(bbox_pred.contiguous(),
                                             reg_offset.contiguous())
            bbox_preds_new.append(bbox_pred)
        return tuple(cls_scores), tuple(bbox_preds_new)

    def deform_sampling(self, x: Tensor, offset: Tensor) -> Tensor:
        """Sampling the feature x according to offset.

        Args:
            x (Tensor): Feature
            offset (Tensor): Spatial offset for for feature sampling

        Returns:
            Tensor: An equivalent implementation of the bnlinear
                interpolation
        """
        b, c, h, w = x.shape
        weight = x.new_ones(c, 1, 1, 1)
        return DeformConv2d(x, offset, weight, 1, 0, 1, c, c)

    def centerness_target(self, anchors: Tensor) -> Tensor:
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            alignment_metrics: Tensor,
                            stride: Tuple[int]) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            alignment_metrics (Tensor): Alignment metrics with shape
                (N, num_total_anchors).
            stride (tuple[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        alignment_metrics = alignment_metrics.reshape(-1, 4)

        targets = labels
        cls_loss_func = self.loss_cls
        loss_cls = cls_loss_func(
            cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_id = self.num_classes
        pos_ids = ((labels >= 0) & (labels < bg_class_id)).nonzero().squeeze(1)
        if len(pos_ids) > 0:
            pos_bbox_targets = bbox_targets[pos_ids]
            pos_bbox_pred = bbox_pred[pos_ids]
            pos_anchors = anchors[pos_ids]
            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]

            # regression loss
            pos_bbox_weight = alignment_metrics[pos_ids]

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, alignment_metrics.sum(
        ), pos_bbox_weight.sum()

    # force_fp32(apply_to=('cls_scores, 'bbox_preds'))
    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     return_targets_only: bool = False) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_gt_bboxes = [
        ]  # (list[Tensor]): ground truth bboxes for each image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        batch_gt_labels = [
        ]  # (list[Tensor]): class indices corresponding to each box
        for gt_instance in batch_gt_instances:
            batch_gt_bboxes.append(gt_instance['bboxes'])
            batch_gt_labels.append(gt_instance['labels'])
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) * stride[0]
            for bbox_pred, stride in zip(bbox_preds,
                                         self.prior_generator.strides)
        ], 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bbox_preds,
            anchor_list,
            valid_flag_list,
            batch_gt_bboxes,
            batch_img_metas,
            gt_bboxes_ignore_list=batch_gt_instances_ignore,
            gt_labels_list=batch_gt_labels,
            label_channels=label_channels)
        if return_targets_only:
            return cls_reg_targets

    def _get_bboxes_single(self,
                           cls_score_list: list[Tensor],
                           bbox_pred_list: list[Tensor],
                           score_factor_list: list[Tensor],
                           mlvl_priors: list[Tensor],
                           img_meta: dict,
                           cfg: ConfigType,
                           rescale: bool = False,
                           with_nms: bool = True,
                           **kwargs) -> Tuple[Tensor]:
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """

        cfg = self.test_cfg if cfg is not None else cfg
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for cls_score, bbox_pred, priors, stride in zip(
                cls_score_list, bbox_pred_list, mlvl_priors,
                self.prior_generator.strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4) * stride[0]
            scores = cls_score.permute(1, 2,
                                       0).reshape(-1, self.cls_out_channels)

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(scores, cfg.score_thr, nms_pre,
                                             dict(bbox_pred, priors))
            scores, labels, kept_anchor_idxs, filtered_results = results
            bboxes = filtered_results['bbox_pred']

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, **kwargs)

    def get_targets(self,
                    cls_scores: Tensor,
                    bbox_preds: Tensor,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    label_channels: int = 1,
                    unmap_outputs: bool = True) -> tuple:
        """Get targets for FAM3D head.

        Args:
            cls_scores (Tensor): Classification predictions of images,
                a 3D-Tensor with shape [num_imgs, num_priors, num_classes].
            bbox_preds (Tensor): Decoded bboxes predictions of one image,
                a 3D-Tensor with shape [num_imgs, num_priors, 4] in [tl_x,
                tl_y, br_x, br_y] format.
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: a tuple contains learning targets.
                anchors_list (list[list[Tensor]]): Anchors of each level.
                labels_list (list[Tensor]): Labels of each level.
                label_weights_list (list[Tensor]): Label weights of each level.
                bbox_targets_list (list[Tensor]): BBox targets of each level.
                norm_alignment_metrics_list (list[Tensor]): Normalized
                    alignment metrics of each level.
        """
        num_imgs = len(batch_gt_instances)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        if batch_gt_instances is None:
            batch_gt_instances = [None] * num_imgs
        # anchor_list: list(b * [-1, 4])
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_assign_metrics, gt_cost_list) = multi_apply(
             self._get_targets_single,
             cls_scores,
             bbox_preds,
             anchor_list,
             valid_flag_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None

        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        norm_alignment_metrics_list = images_to_levels(all_assign_metrics,
                                                       num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, norm_alignment_metrics_list, gt_cost_list)

    def _get_targets_single(self,
                            cls_scores: List[Tensor],
                            bbox_preds: List[Tensor],
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: InstanceData | None = None,
                            label_channel: int = 1,
                            unmap_outputs: bool = True) -> Tensor:
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (list(Tensor)): Box scores for each image.
            bbox_preds (list(Tensor)): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of shape (num_anchors,).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                nchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                norm_alignment_metrics (Tensor): Normalized alignment metrics
                    of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7

        gt_bboxes = gt_instances['bboxes']
        gt_labels = gt_instances['labels']
        gt_bboxes_ignore = gt_instances_ignore['bboxes']

        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        # TODO: use point generator
        ct_priors = self.centerness_target(anchors)
        strides = (anchors[..., 2:] -
                   anchors[..., 2]) / 8  # octave_based_scale
        ct_priors = torch.cat([ct_priors, strides], dim=1)

        assign_result = self.alignment_assigner.assign(
            cls_scores[inside_flags, :], bbox_preds[inside_flags, :],
            ct_priors, gt_bboxes, gt_bboxes_ignore, gt_labels, self.alpha,
            self.beta)
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics
        gt_cost = assign_result.gt_cost

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        norm_alignment_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            norm_alignment_metrics = unmap(norm_alignment_metrics,
                                           num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets,
                norm_alignment_metrics, gt_cost)
