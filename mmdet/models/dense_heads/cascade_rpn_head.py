from __future__ import division

import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (RegionAssigner, build_assigner, build_sampler,
                        images_to_levels, multi_apply)
from mmdet.ops import DeformConv, nms
from ..builder import HEADS, build_head
from .anchor_head import AnchorHead


class AdaptiveConv(nn.Module):
    """ AdaptiveConv used to adapt the sampling location with the anchors.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        adapt_cfg (dict): adaptation type and arguments.
            dilation (uniform anchor) or offset (arbitrary anchors).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 adapt_cfg=dict(type='dilation', dilation=3)):
        super(AdaptiveConv, self).__init__()
        self.adapt_cfg = adapt_cfg
        assert isinstance(adapt_cfg, dict) and 'type' in adapt_cfg

        self.adapt_type = adapt_cfg['type']
        assert self.adapt_type in ['offset', 'dilation']

        if self.adapt_type == 'offset':
            self.conv = DeformConv(in_channels, out_channels, 3, padding=1)
        else:
            dilation = adapt_cfg.get('dilation', 3)
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation)

    def init_weights(self):
        normal_init(self.conv, std=0.01)

    def forward(self, x, offset):
        if self.adapt_type == 'offset':
            N, _, H, W = x.shape
            assert offset is not None
            assert H * W == offset.shape[1]
            # reshape [N, NA, 18] to (N, 18, H, W)
            offset = offset.permute(0, 2, 1).reshape(N, -1, H, W)
            x = self.conv(x, offset)
        else:
            assert offset is None
            x = self.conv(x)
        return x


@HEADS.register_module()
class StageCascadeRPNHead(AnchorHead):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_generator (dict): anchor generator config.
        adapt_cfg (dict): adaptation config.
        bridged_feature (bool): wheater update rpn feature.
        with_cls (bool): wheather use cls branch.
        sampling (bool): wheather use sampling.
    """

    def __init__(self,
                 in_channels,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8],
                     ratios=[1.0],
                     strides=[4, 8, 16, 32, 64]),
                 adapt_cfg=dict(type='dilation', dilation=3),
                 bridged_feature=False,
                 with_cls=True,
                 sampling=True,
                 **kwargs):
        self.with_cls = with_cls
        self.anchor_strides = anchor_generator['strides']
        self.anchor_scales = anchor_generator['scales']
        self.bridged_feature = bridged_feature
        self.adapt_cfg = adapt_cfg
        super(StageCascadeRPNHead, self).__init__(
            1,
            in_channels,
            background_label=0,
            anchor_generator=anchor_generator,
            **kwargs)

        # override sampling and sampler
        self.sampling = sampling
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self):
        self.rpn_conv = AdaptiveConv(self.in_channels, self.feat_channels,
                                     self.adapt_cfg)
        if self.with_cls:
            self.rpn_cls = nn.Conv2d(self.feat_channels,
                                     self.num_anchors * self.cls_out_channels,
                                     1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        self.rpn_conv.init_weights()
        normal_init(self.rpn_reg, std=0.01)
        if self.with_cls:
            normal_init(self.rpn_cls, std=0.01)

    def forward_single(self, x, offset):
        bridged_x = x
        x = self.relu(self.rpn_conv(x, offset))
        if self.bridged_feature:
            bridged_x = x  # update feature
        cls_score = self.rpn_cls(x) if self.with_cls else None
        bbox_pred = self.rpn_reg(x)
        return bridged_x, cls_score, bbox_pred

    def forward(self, feats, offset_list=None):
        if offset_list is None:
            offset_list = [None for _ in range(len(feats))]
        return multi_apply(self.forward_single, feats, offset_list)

    def _region_targets_single(self,
                               anchors,
                               valid_flags,
                               gt_bboxes,
                               gt_bboxes_ignore,
                               gt_labels,
                               img_meta,
                               featmap_sizes,
                               label_channels=1):
        assign_result = self.assigner.assign(
            anchors,
            valid_flags,
            gt_bboxes,
            img_meta,
            featmap_sizes,
            self.anchor_scales[0],
            self.anchor_strides,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_labels=None,
            allowed_border=self.train_cfg.allowed_border)
        flat_anchors = torch.cat(anchors)
        sampling_result = self.sampler.sample(assign_result, flat_anchors,
                                              gt_bboxes)

        num_anchors = flat_anchors.shape[0]
        bbox_targets = torch.zeros_like(flat_anchors)
        bbox_weights = torch.zeros_like(flat_anchors)
        labels = flat_anchors.new_zeros(num_anchors, dtype=torch.long)
        label_weights = flat_anchors.new_zeros(num_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def region_targets(self,
                       anchor_list,
                       valid_flag_list,
                       gt_bboxes_list,
                       img_metas,
                       featmap_sizes,
                       gt_bboxes_ignore_list=None,
                       gt_labels_list=None,
                       label_channels=1,
                       unmap_outputs=True):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            featmap_sizes (list[Tensor]): Feature mapsize each level
            gt_bboxes_ignore_list (list[Tensor]): Ignore bboxes of each images
            gt_labels_list (list[Tensor]): Ground truth labels of each image
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            region_targets (tuple)
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._region_targets_single,
             anchor_list,
             valid_flag_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             featmap_sizes=featmap_sizes,
             label_channels=label_channels)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes,
                    img_metas,
                    featmap_sizes,
                    gt_bboxes_ignore=None,
                    label_channels=1):
        if isinstance(self.assigner, RegionAssigner):
            cls_reg_targets = self.region_targets(
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                featmap_sizes,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                label_channels=label_channels)
        else:
            cls_reg_targets = super(StageCascadeRPNHead, self).get_targets(
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                label_channels=label_channels)
        return cls_reg_targets

    def anchor_offset(self, anchor_list, anchor_strides, featmap_sizes):
        """ Get offest for deformable conv based on anchor shape
        NOTE: currently support deformable kernel_size=3 and dilation=1

        Args:
            anchor_list (list[list[tensor])): [NI, NLVL, NA, 4] list of
                multi-level anchors
                anchor_strides (list): anchor stride of each level
        Returns:
            offset_list (list[tensor]): [NLVL, NA, 2, 18]: offset of DeformConv
                kernel.
        """

        def _shape_offset(anchors, stride):
            # currently support kernel_size=3 and dilation=1
            ks = 3
            dilation = 1
            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            yy, xx = torch.meshgrid(idx, idx)  # return order matters
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            w = (anchors[:, 2] - anchors[:, 0] + 1) / stride
            h = (anchors[:, 3] - anchors[:, 1] + 1) / stride
            w = w / (ks - 1) - dilation
            h = h / (ks - 1) - dilation
            offset_x = w[:, None] * xx  # (NA, ks**2)
            offset_y = h[:, None] * yy  # (NA, ks**2)
            return offset_x, offset_y

        def _ctr_offset(anchors, stride, featmap_size):
            feat_h, feat_w = featmap_size
            assert len(anchors) == feat_h * feat_w

            x = (anchors[:, 0] + anchors[:, 2]) * 0.5
            y = (anchors[:, 1] + anchors[:, 3]) * 0.5
            # compute centers on feature map
            x = (x - (stride - 1) * 0.5) / stride
            y = (y - (stride - 1) * 0.5) / stride
            # compute predefine centers
            xx = torch.arange(0, feat_w, device=anchors.device)
            yy = torch.arange(0, feat_h, device=anchors.device)
            yy, xx = torch.meshgrid(yy, xx)
            xx = xx.reshape(-1).type_as(x)
            yy = yy.reshape(-1).type_as(y)

            offset_x = x - xx  # (NA, )
            offset_y = y - yy  # (NA, )
            return offset_x, offset_y

        num_imgs = len(anchor_list)
        num_lvls = len(anchor_list[0])
        dtype = anchor_list[0][0].dtype
        device = anchor_list[0][0].device
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        offset_list = []
        for i in range(num_imgs):
            mlvl_offset = []
            for lvl in range(num_lvls):
                c_offset_x, c_offset_y = _ctr_offset(anchor_list[i][lvl],
                                                     anchor_strides[lvl],
                                                     featmap_sizes[lvl])
                s_offset_x, s_offset_y = _shape_offset(anchor_list[i][lvl],
                                                       anchor_strides[lvl])

                # offset = ctr_offset + shape_offset
                offset_x = s_offset_x + c_offset_x[:, None]
                offset_y = s_offset_y + c_offset_y[:, None]

                # offset order (y0, x0, y1, x2, .., y8, x8, y9, x9)
                offset = torch.stack([offset_y, offset_x], dim=-1)
                offset = offset.reshape(offset.size(0), -1)  # [NA, 2*ks**2]
                mlvl_offset.append(offset)
            offset_list.append(torch.cat(mlvl_offset))  # [totalNA, 2*ks**2]
        offset_list = images_to_levels(offset_list, num_level_anchors)
        return offset_list

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # classification loss
        if self.with_cls:
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(-1, self.cls_out_channels)
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_reg = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        if self.with_cls:
            return loss_cls, loss_reg
        return None, loss_reg

    def loss(self,
             anchor_list,
             valid_flag_list,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            featmap_sizes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        if self.sampling:
            num_total_samples = num_total_pos + num_total_neg
        else:
            num_total_samples = sum([label.numel()
                                     for label in labels_list]) / 200.0

        # change per image, per level anchor_list to per_level, per_image
        mlvl_anchor_list = list(zip(*anchor_list))
        # concat mlvl_anchor_list
        mlvl_anchor_list = [
            torch.cat(anchors, dim=0) for anchors in mlvl_anchor_list
        ]

        losses = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            mlvl_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        if self.with_cls:
            return dict(loss_rpn_cls=losses[0], loss_rpn_reg=losses[1])
        return dict(loss_rpn_reg=losses[1])

    def get_bboxes(self,
                   anchor_list,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                anchor_list[img_id], img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def refine_bboxes(self, anchor_list, bbox_preds, img_metas):
        num_levels = len(bbox_preds)
        new_anchor_list = []
        for img_id in range(len(img_metas)):
            mlvl_anchors = []
            for i in range(num_levels):
                bbox_pred = bbox_preds[i][img_id].detach()
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                img_shape = img_metas[img_id]['img_shape']
                bboxes = self.bbox_coder.decode(anchor_list[img_id][i],
                                                bbox_pred, img_shape)
                mlvl_anchors.append(bboxes)
            new_anchor_list.append(mlvl_anchors)
        return new_anchor_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = self.bbox_coder.decode(anchors, rpn_bbox_pred,
                                               img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals


@HEADS.register_module()
class CascadeRPNHead(nn.Module):  # TODO: inherit BaseDenseHead

    def __init__(self, num_stages, stages, train_cfg, test_cfg):
        super(CascadeRPNHead, self).__init__()
        assert num_stages == len(stages)
        self.num_stages = num_stages
        self.stages = nn.ModuleList()
        for i in range(len(stages)):
            train_cfg_i = train_cfg[i] if train_cfg is not None else None
            stages[i].update(train_cfg=train_cfg_i)
            stages[i].update(test_cfg=test_cfg)
            self.stages.append(build_head(stages[i]))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self):
        for i in range(self.num_stages):
            self.stages[i].init_weights()

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None):
        assert gt_labels is None, 'RPN does not require gt_labels'

        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        anchor_list, valid_flag_list = self.stages[0].get_anchors(
            featmap_sizes, img_metas)

        losses = dict()

        for i in range(self.num_stages):
            stage = self.stages[i]

            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list,
                                                  stage.anchor_strides,
                                                  featmap_sizes)
            else:
                offset_list = None
            x, cls_score, bbox_pred = stage(x, offset_list)
            rpn_loss_inputs = (anchor_list, valid_flag_list, cls_score,
                               bbox_pred, gt_bboxes, img_metas)
            stage_loss = stage.loss(*rpn_loss_inputs)
            for name, value in stage_loss.items():
                losses['s{}.{}'.format(i, name)] = value

            # refine boxes
            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred,
                                                  img_metas)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.rpn_head[-1].get_bboxes(
                anchor_list, cls_score, bbox_pred, img_metas, self.test_cfg)
            return losses, proposal_list

    def simple_test_rpn(self, x, img_metas):
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        anchor_list, _ = self.stages[0].get_anchors(featmap_sizes, img_metas)

        for i in range(self.num_stages):
            stage = self.stages[i]
            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list,
                                                  stage.anchor_strides,
                                                  featmap_sizes)
            else:
                offset_list = None
            x, cls_score, bbox_pred = stage(x, offset_list)
            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred,
                                                  img_metas)

        proposal_list = self.stages[-1].get_bboxes(anchor_list, cls_score,
                                                   bbox_pred, img_metas,
                                                   self.test_cfg)
        return proposal_list

    def aug_test_rpn(self, x, img_metas):
        raise NotImplementedError
