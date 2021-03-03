import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, bias_init_with_prob, constant_init,
                      normal_init)
from mmcv.runner import force_fp32
from mmcv.ops.nms import batched_nms

from mmdet.core import (build_assigner, build_sampler,
                        multiclass_nms)
from .anchor_head import AnchorHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class TinaFaceHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_iou=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(TinaFaceHead, self).__init__(num_classes,in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_centerness = build_loss(loss_iou)

    def _init_layers(self):
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
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.retina_iou = nn.Conv2d(
            self.feat_channels, self.num_anchors, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        constant_init(self.retina_reg, 0.)
        constant_init(self.retina_iou, 0.)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        iou_pred = self.retina_iou(reg_feat)

        return cls_score, bbox_pred, iou_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   iou,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            iou (list[Tensor]): iou for each scale level with
                shape (N, num_anchors * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                iou[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale,
                                                with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           iou,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            iou (list[Tensor]): iou for a single scale level
                with shape (num_anchors * 1, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        alpha = 0.4
        height_th = 9

        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        # mlvl_centerness = []
        for cls_score, bbox_pred, centerness, anchors in zip(
                cls_scores, bbox_preds, iou, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permute(1, 2, 0).reshape(-1, 1).sigmoid()

            if alpha is None:
                scores *= centerness
            elif isinstance(alpha, float):
                scores = torch.pow(scores, 2 * alpha) * torch.pow(
                    centerness, 2 * (1 - alpha))
            else:
                raise ValueError("alpha must be float or None")

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                # centerness = centerness[topk_inds]

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            # mlvl_centerness.append(centerness)

        def filter_boxes(boxes, min_scale, max_scale):
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            scales = torch.sqrt(ws * hs)

            return (scales >= max(1, min_scale)) & (scales <= max_scale)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)

        keeps = filter_boxes(mlvl_bboxes, 1, 10000)
        mlvl_bboxes = mlvl_bboxes[keeps]
        mlvl_scores = mlvl_scores[keeps]

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        # mlvl_centerness = torch.cat(mlvl_centerness)

        if height_th is not None:
            hs = mlvl_bboxes[:, 3] - mlvl_bboxes[:, 1]
            valid = (hs >= height_th)
            mlvl_bboxes, mlvl_scores = (
                mlvl_bboxes[valid], mlvl_scores[valid])

        if with_nms:
            det_bboxes, det_labels = self._lb_multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=None)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def _lb_multiclass_nms(self,multi_bboxes,
                           multi_scores,
                           score_thr,
                           nms_cfg,
                           max_num=-1,
                           score_factors=None):
        """NMS for multi-class bboxes.

        Args:
            multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
            multi_scores (Tensor): shape (n, #class), where the last column
                contains scores of the background class, but this will be ignored.
            score_thr (float): bbox threshold, bboxes with scores lower than it
                will not be considered.
            nms_thr (float): NMS IoU threshold
            max_num (int): if there are more than max_num bboxes after NMS,
                only top max_num will be kept.
            score_factors (Tensor): The factors multiplied to scores before
                applying NMS

        Returns:
            tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
                are 0-based.
        """
        num_classes = multi_scores.size(1) - 1
        # exclude background category
        if multi_bboxes.shape[1] > 4:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, 4)
        scores = multi_scores[:, :-1]

        # filter out boxes with low scores
        valid_mask = scores > score_thr

        # We use masked_select for ONNX exporting purpose,
        # which is equivalent to bboxes = bboxes[valid_mask]
        # (TODO): as ONNX does not support repeat now,
        # we have to use this ugly code
        bboxes = torch.masked_select(
            bboxes,
            torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                        -1)).view(-1, 4)
        if score_factors is not None:
            scores = scores * score_factors[:, None]
        scores = torch.masked_select(scores, valid_mask)
        labels = valid_mask.nonzero()[:, 1]

        if bboxes.numel() == 0:
            bboxes = multi_bboxes.new_zeros((0, 5))
            labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

            if torch.onnx.is_in_onnx_export():
                raise RuntimeError('[ONNX Error] Can not record NMS '
                                   'as it has not been executed this time')
            return bboxes, labels

        inds = scores.argsort(descending=True)
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]

        batch_bboxes = torch.empty((0, 4),
                                   dtype=bboxes.dtype,
                                   device=bboxes.device)
        batch_scores = torch.empty((0,), dtype=scores.dtype, device=scores.device)
        batch_labels = torch.empty((0,), dtype=labels.dtype, device=labels.device)
        while bboxes.shape[0] > 0:
            num = min(100000, bboxes.shape[0])
            batch_bboxes = torch.cat([batch_bboxes, bboxes[:num]])
            batch_scores = torch.cat([batch_scores, scores[:num]])
            batch_labels = torch.cat([batch_labels, labels[:num]])
            bboxes = bboxes[num:]
            scores = scores[num:]
            labels = labels[num:]

            _, keep = batched_nms(batch_bboxes, batch_scores, batch_labels,
                                  nms_cfg)
            batch_bboxes = batch_bboxes[keep]
            batch_scores = batch_scores[keep]
            batch_labels = batch_labels[keep]

        dets = torch.cat([batch_bboxes, batch_scores[:, None]], dim=-1)
        labels = batch_labels

        if max_num > 0:
            dets = dets[:max_num]
            labels = labels[:max_num]

        return dets, labels
