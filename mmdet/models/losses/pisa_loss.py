import torch

from mmdet.core import bbox_overlaps


def isr_p(cls_score,
          bbox_pred,
          bbox_targets,
          rois,
          sampling_results,
          loss_cls,
          bbox_coder,
          k=2,
          bias=0):
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets
    pos_label_inds = ((labels >= 0) & (labels < 80)).nonzero().reshape(-1)
    pos_labels = labels[pos_label_inds]

    # if no positive samples, return the original targets
    num_pos = float(pos_label_inds.size(0))
    if num_pos == 0:
        return bbox_targets

    # merge pos_assigned_gt_inds of per image to a single tensor
    gts = list()
    last_max_gt = 0
    for i in range(len(sampling_results)):
        gt_i = sampling_results[i].pos_assigned_gt_inds
        gts.append(gt_i + last_max_gt)
        last_max_gt = gt_i.max() + 1
    gts = torch.cat(gts)
    assert len(gts) == num_pos

    if rois.size(-1) == 5:
        pos_rois = rois[pos_label_inds][:, 1:]
    else:
        pos_rois = rois[pos_label_inds]

    if bbox_pred.size(-1) > 4:
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)
        pos_delta_pred = bbox_pred[pos_label_inds, pos_labels].view(-1, 4)
    else:
        pos_delta_pred = bbox_pred[pos_label_inds].view(-1, 4)

    # compute iou of the predicted bbox and the corresponding GT
    pos_delta_target = bbox_targets[pos_label_inds].view(-1, 4)
    pos_bbox_pred = bbox_coder.decode(pos_rois, pos_delta_pred)
    target_bbox_pred = bbox_coder.decode(pos_rois, pos_delta_target)
    ious = bbox_overlaps(pos_bbox_pred, target_bbox_pred, is_aligned=True)

    pos_imp_weights = label_weights[pos_label_inds]
    # Two steps to compute IoU-HLR. Samples are first sorted by IoU locally,
    # then sorted again within the same-rank group
    max_l_num = pos_labels.bincount().max()
    for l in pos_labels.unique():
        l_inds = (pos_labels == l).nonzero().view(-1)
        l_gts = gts[l_inds]
        for t in l_gts.unique():
            t_inds = l_inds[l_gts == t]
            t_ious = ious[t_inds]
            _, t_iou_rank_idx = t_ious.sort(descending=True)
            _, t_iou_rank = t_iou_rank_idx.sort()
            ious[t_inds] += max_l_num - t_iou_rank.float()
        l_ious = ious[l_inds]
        _, l_iou_rank_idx = l_ious.sort(descending=True)
        _, l_iou_rank = l_iou_rank_idx.sort()  # IoU-HLR
        # linearly map HLR to label weights
        pos_imp_weights[l_inds] *= (max_l_num - l_iou_rank.float()) / max_l_num

    pos_imp_weights = (bias + pos_imp_weights * (1 - bias)).pow(k)

    # normalize to make the new weighted loss value equal to the original loss
    pos_loss_cls = loss_cls(
        cls_score[pos_label_inds], pos_labels, reduction_override='none')
    if pos_loss_cls.dim() > 1:
        ori_pos_loss_cls = pos_loss_cls * label_weights[pos_label_inds][:,
                                                                        None]
        new_pos_loss_cls = pos_loss_cls * pos_imp_weights[:, None]
    else:
        ori_pos_loss_cls = pos_loss_cls * label_weights[pos_label_inds]
        new_pos_loss_cls = pos_loss_cls * pos_imp_weights
    pos_loss_cls_ratio = ori_pos_loss_cls.sum() / new_pos_loss_cls.sum()
    pos_imp_weights = pos_imp_weights * pos_loss_cls_ratio
    label_weights[pos_label_inds] = pos_imp_weights

    bbox_targets = labels, label_weights, bbox_targets, bbox_weights
    return bbox_targets


def carl_loss(cls_score,
              labels,
              bbox_pred,
              bbox_targets,
              loss_bbox,
              k=1,
              bias=0.2,
              avg_factor=None,
              sigmoid=False):
    pos_label_inds = ((labels >= 0) & (labels < 80)).nonzero().reshape(-1)
    pos_labels = labels[pos_label_inds]

    # multiply pos_cls_score with the corresponding bbox weight
    # and remain gradient
    if sigmoid:
        pos_cls_score = cls_score.sigmoid()[pos_label_inds, pos_labels]
    else:
        pos_cls_score = cls_score.softmax(-1)[pos_label_inds, pos_labels]
    carl_loss_weights = (bias + (1 - bias) * pos_cls_score).pow(k)

    # normalize carl_loss_weight to make its sum equal to num positive
    num_pos = float(pos_cls_score.size(0))
    weight_ratio = num_pos / carl_loss_weights.sum()
    carl_loss_weights *= weight_ratio

    if avg_factor is None:
        avg_factor = bbox_targets.size(0)
    if bbox_pred.size(-1) > 4:
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)
        pos_bbox_preds = bbox_pred[pos_label_inds, pos_labels]
    else:
        pos_bbox_preds = bbox_pred[pos_label_inds]
    ori_loss_reg = loss_bbox(
        pos_bbox_preds,
        bbox_targets[pos_label_inds],
        reduction_override='none') / avg_factor
    carl_loss_weights = carl_loss_weights[:, None].expand(-1, 4)
    loss_carl = (ori_loss_reg * carl_loss_weights).sum()
    return dict(loss_carl=loss_carl[None])
