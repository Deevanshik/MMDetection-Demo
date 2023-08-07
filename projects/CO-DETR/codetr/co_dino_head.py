# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 
from typing import Dict, List, Tuple
from torch import Tensor
from mmengine.structures import InstanceData
from mmcv.ops import batched_nms
from mmdet.utils import InstanceList, reduce_mean
from mmdet.structures import SampleList
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps)
from mmdet.models.utils import multi_apply, unpack_gt_instances
from mmdet.models.layers.transformer import inverse_sigmoid
from mmcv.ops import batched_nms
from mmdet.models.task_modules.samplers import PseudoSampler
from mmcv.cnn import Linear

from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmdet.models.layers import CdnQueryGenerator
from mmdet.models import DINOHead


@MODELS.register_module()
class CoDINOHead(DINOHead):

    def __init__(self,
                 *args,
                 num_query=900,
                 transformer=None,
                 in_channels=2048,
                 max_pos_coords=300,
                 lambda_1=1,
                 dn_cfg=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                **kwargs):
        self.with_box_refine=True
        self.mixed_selection=True
        self.with_attn_mask=False
        self.in_channels=in_channels
        self.max_pos_coords = max_pos_coords
        self.lambda_1 = lambda_1
        self.positional_encoding = positional_encoding
        self.num_query=num_query

        if 'two_stage_num_proposals' in transformer:
            assert transformer['two_stage_num_proposals'] == num_query, \
                'two_stage_num_proposals must be equal to num_query for DINO'
        else:
            transformer['two_stage_num_proposals'] = num_query
        transformer['as_two_stage'] = True
        if self.mixed_selection:
            transformer['mixed_selection'] = self.mixed_selection
        self.transformer = transformer
        self.act_cfg = transformer.get('act_cfg', dict(type='ReLU', inplace=True))

        super().__init__(*args, **kwargs)

        self.activate = MODELS.build(self.act_cfg)
        self.positional_encoding = MODELS.build(self.positional_encoding)
        self.init_denoising(dn_cfg)
    
    def _init_layers(self):
        self.transformer = MODELS.build(self.transformer)
        self.embed_dims = self.transformer.embed_dims
        assert hasattr( self.positional_encoding, 'num_feats')
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
    
        """Initialize classification branch and regression branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)

        self.query_embedding = None
        # NOTE The original repo of DINO set the num_embeddings 92 for coco,
        # 91 (0~90) of which represents target classes and the 92 (91)
        # indicates [Unknown] class. However, the embedding of unknown class
        # is not used in the original DINO
        self.label_embedding = nn.Embedding(self.cls_out_channels,
                                            self.embed_dims)
        self.downsample = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, self.embed_dims)
        )

    def init_denoising(self, dn_cfg):
        if dn_cfg is not None:
            dn_cfg['num_classes'] = self.num_classes
            dn_cfg['num_matching_queries'] = self.num_query
            dn_cfg['embed_dims'] = self.embed_dims
        self.dn_generator = CdnQueryGenerator(**dn_cfg)

    def forward(self,
                mlvl_feats,
                img_metas,
                dn_label_query=None,
                dn_bbox_query=None,
                attn_mask=None):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        hs, inter_references, topk_score, topk_anchor, enc_outputs = \
            self.transformer(
                mlvl_feats,
                mlvl_masks,
                query_embeds,
                mlvl_positional_encodings,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )
        outs = []
        num_level = len(mlvl_feats)
        start = 0
        for lvl in range(num_level):
            bs, c, h, w = mlvl_feats[lvl].shape
            end = start + h*w
            feat = enc_outputs[start:end].permute(1, 2, 0).contiguous()
            start = end
            outs.append(feat.reshape(bs, c, h, w))
        outs.append(self.downsample(outs[-1]))

        hs = hs.permute(0, 2, 1, 3)

        if dn_label_query is not None and dn_label_query.size(1) == 0:
            # NOTE: If there is no target in the image, the parameters of
            # label_embedding won't be used in producing loss, which raises
            # RuntimeError when using distributed mode.
            hs[0] += self.label_embedding.weight[0, 0] * 0.0

        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference, eps=1e-3)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        return outputs_classes, outputs_coords, topk_score, topk_anchor, outs
    
    def predict(self,
                feats: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network. Over-write
        because img_metas are needed as inputs for bbox_head.

        Args:
            hidden_states (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self.forward(feats, batch_img_metas)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)

        return predictions

    def predict_by_feat(self,
                        all_cls_scores,
                        all_bbox_preds,
                        enc_cls_scores,
                        enc_bbox_preds,
                        enc_outputs,
                        batch_img_metas,
                        rescale=True):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        score_thr = self.test_cfg.get('score_thr', 0)
        with_nms = self.test_cfg.get('nms', None)

        img_shape = img_meta['img_shape']
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]
        
        if score_thr > 0:
            valid_mask = scores > score_thr
            scores = scores[valid_mask]
            bbox_pred = bbox_pred[valid_mask]
            det_labels = det_labels[valid_mask]
        
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels

        if with_nms and results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(results.bboxes, 
                                                results.scores,
                                                results.labels, 
                                                self.test_cfg.nms)
            results = results[keep_idxs]
            results.scores = det_bboxes[:, -1]
            results = results[:max_per_img]

        return results
    
    def loss(self, x, batch_data_samples):
        assert self.dn_generator is not None, '"dn_cfg" must be set'

        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
        
        dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
            self.dn_generator(batch_data_samples)
        
        outs = self(x, batch_img_metas, dn_label_query, dn_bbox_query, attn_mask)

        loss_inputs = outs[:-1] + (batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        enc_outputs = outs[-1]
        return losses, enc_outputs

    def forward_aux(self, mlvl_feats, img_metas, aux_targets, head_idx):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """
        aux_coords, aux_labels, aux_targets, aux_label_weights, aux_bbox_weights, aux_feats, attn_masks = aux_targets
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        hs, inter_references = self.transformer.forward_aux(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    aux_coords,
                    pos_feats=aux_feats,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
                    return_encoder_output=True,
                    attn_masks=attn_masks,
                    head_idx=head_idx
            )

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference, eps=1e-3)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        return outputs_classes, outputs_coords, \
                None, None

    def loss_single_aux(self,
                        cls_scores,
                        bbox_preds,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        img_metas,
                        gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        num_q = cls_scores.size(1)
        try:
            labels = labels.reshape(num_imgs * num_q)
            label_weights = label_weights.reshape(num_imgs * num_q)
            bbox_targets = bbox_targets.reshape(num_imgs * num_q, 4)
            bbox_weights = bbox_weights.reshape(num_imgs * num_q, 4)
        except:
            return cls_scores.mean()*0, cls_scores.mean()*0, cls_scores.mean()*0

        bg_class_ind = self.num_classes
        num_total_pos = len(((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1))
        num_total_neg = num_imgs*num_q - num_total_pos

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        scores = label_weights.new_zeros(labels.shape)
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
        pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
        pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
        scores[pos_inds] = bbox_overlaps(
            pos_decode_bbox_pred.detach(),
            pos_decode_bbox_targets,
            is_aligned=True)
        loss_cls = self.loss_cls(
            cls_scores, (labels, scores),
            weight=label_weights,
            avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls*self.lambda_1, loss_bbox*self.lambda_1, loss_iou*self.lambda_1
