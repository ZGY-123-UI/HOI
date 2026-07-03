import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import accuracy, get_world_size, is_dist_avail_and_initialized
import numpy as np


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1-1e-4)
    return y

class SetCriterionHOI(nn.Module):

    def __init__(self, cfg, matcher, weight_dict, losses):
        super().__init__()
        self.num_obj_classes = cfg.INPUT.NUM_OBJ_CLASSES
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        empty_weight = torch.ones(self.num_obj_classes + 1, dtype=torch.bfloat16)
        empty_weight[-1] = cfg.LOSS.COEF_CE_LOSS_OBJ
        self.register_buffer('empty_weight', empty_weight)
        self.alpha = cfg.LOSS.ALPHA_FOCAL_LOSS_HOI

        clip_soft_cfg = cfg.LOSS.get("CLIP_SOFT_LABEL", {})
        self.use_clip_soft_label = clip_soft_cfg.get("ENABLED", False)
        self.clip_soft_mix_alpha = clip_soft_cfg.get("MIX_ALPHA", 0.0)
        self.clip_soft_temperature = clip_soft_cfg.get("TEMPERATURE", 1.0)
        if self.use_clip_soft_label:
            clip_soft_path = clip_soft_cfg.get("PATH", "")
            if not clip_soft_path:
                raise ValueError("LOSS.CLIP_SOFT_LABEL.PATH must be set when CLIP soft label is enabled")
            soft_data = torch.load(clip_soft_path, map_location="cpu", weights_only=True)
            if "clip_soft_label" not in soft_data:
                raise KeyError(f"clip soft label file must contain 'clip_soft_label': {clip_soft_path}")
            self.register_buffer("clip_soft_label", soft_data["clip_soft_label"].float())
        else:
            self.clip_soft_label = None

        prior_cfg = cfg.LOSS.get("HOI_PRIOR", {})
        self.use_hoi_prior = prior_cfg.get("ENABLED", False)

        union_clip_cfg = cfg.LOSS.get("UNION_CROP_CLIP_KD", {})
        self.use_union_clip_kd = union_clip_cfg.get("ENABLED", False)
        self.union_clip_temperature = union_clip_cfg.get("TEMPERATURE", 2.0)

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.inference_mode()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_hoi_labels(self, outputs, targets, indices, num_interactions, topk=5, log=True):
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes_o = target_classes_o.to(src_logits.dtype)
        target_classes = torch.zeros_like(src_logits)
        matched_train_targets = target_classes_o

        if self.use_clip_soft_label:
            clip_soft_label = self._get_clip_soft_label(src_logits)
            if target_classes_o.numel() > 0:
                clip_targets = target_classes_o @ clip_soft_label
                clip_targets = torch.maximum(clip_targets, target_classes_o)
                clip_targets = clip_targets.clamp(max=1.0)
                mix_alpha = max(0.0, min(float(self.clip_soft_mix_alpha), 1.0))
                # Stage 1: blend hard HOI labels with CLIP semantic soft labels before
                # the main HOI focal loss, i.e. y_final = (1-a)*y_hard + a*y_clip.
                matched_train_targets = ((1.0 - mix_alpha) * target_classes_o + mix_alpha * clip_targets).clamp(0.0, 1.0)

        target_classes[idx] = matched_train_targets
        src_prob = _sigmoid(src_logits)
        loss_hoi_ce = self._neg_loss(src_prob, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_hoi_labels': loss_hoi_ce}

        if self.use_clip_soft_label and log:
            matched_logits = src_logits[idx]
            if matched_logits.numel() > 0:
                matched_soft_targets = matched_train_targets
                pred_soft = torch.sigmoid(matched_logits / self.clip_soft_temperature)
                loss_clip_soft = F.binary_cross_entropy(pred_soft, matched_soft_targets, reduction="mean")
            else:
                loss_clip_soft = src_logits.sum() * 0.0
            losses["loss_hoi_clip_soft"] = loss_clip_soft

        if self.use_hoi_prior and log:
            if "hoi_prior_mask" not in outputs:
                raise KeyError("HOI prior is enabled but outputs do not contain 'hoi_prior_mask'")
            prior_mask = outputs["hoi_prior_mask"].to(device=src_logits.device, dtype=src_logits.dtype)
            if prior_mask.shape != src_logits.shape:
                raise ValueError(
                    f"HOI prior mask dim mismatch: hoi_prior_mask={prior_mask.shape}, "
                    f"pred_hoi_logits={src_logits.shape}"
                )
            # Stage 2: penalize probability mass assigned to HOI classes that the
            # action-object feasibility bank marks as unlikely for the predicted object.
            losses["loss_hoi_prior"] = (src_prob * (1.0 - prior_mask)).mean()

        if self.use_union_clip_kd and log:
            if "clip_union_teacher_logits" not in outputs:
                raise KeyError("Union crop CLIP KD is enabled but outputs do not contain 'clip_union_teacher_logits'")
            teacher_logits = outputs["clip_union_teacher_logits"].to(device=src_logits.device, dtype=src_logits.dtype)
            if teacher_logits.shape != src_logits.shape:
                raise ValueError(
                    f"Union CLIP teacher dim mismatch: teacher={teacher_logits.shape}, "
                    f"pred_hoi_logits={src_logits.shape}"
                )
            temperature = self.union_clip_temperature
            # Stage 4: distill union-crop CLIP image/text contrastive logits into
            # the student HOI logits without changing the detector matching path.
            loss_union_clip = F.kl_div(
                F.log_softmax(src_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)
            losses["loss_union_clip_kd"] = loss_union_clip

        if "pred_hoi_masked_logits" in outputs:
            masked_logits = outputs["pred_hoi_masked_logits"]
            if masked_logits.shape != src_logits.shape:
                raise ValueError(
                    f"HOI masked semantic logits dim mismatch: masked={masked_logits.shape}, "
                    f"pred_hoi_logits={src_logits.shape}"
                )
            masked_prob = _sigmoid(masked_logits)
            losses["loss_hoi_masked_semantic"] = self._neg_loss(
                masked_prob, target_classes, weights=None, alpha=self.alpha
            )

        if log:
            _, pred = src_prob[idx].topk(topk, 1, True, True)
            acc = 0.0
            for tid, target in enumerate(target_classes_o):
                tgt_idx = torch.where(target == 1)[0]
                if len(tgt_idx) == 0:
                    continue
                acc_pred = 0.0
                for tgt_rel in tgt_idx:
                    acc_pred += (tgt_rel in pred[tid])
                acc += acc_pred / len(tgt_idx)
            rel_labels_error = 100 - 100 * acc / max(len(target_classes_o), 1)
            losses['hoi_class_error'] = torch.from_numpy(np.array(
                rel_labels_error)).to(src_logits.device).float()
        return losses

    def _get_clip_soft_label(self, src_logits):
        clip_soft_label = self.clip_soft_label.to(device=src_logits.device, dtype=src_logits.dtype)
        if clip_soft_label.shape[0] != src_logits.shape[-1] or clip_soft_label.shape[1] != src_logits.shape[-1]:
            raise ValueError(
                f"CLIP soft label dim mismatch: clip_soft_label={clip_soft_label.shape}, "
                f"pred_hoi_logits={src_logits.shape}"
            )
        return clip_soft_label

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        # Stage 1/5: soft labels are fractional, so positives/negatives are
        # weighted by the target value instead of using only gt == 1 masks.
        pos_inds = gt.float()
        neg_inds = (1.0 - gt).clamp(min=0.0)

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        if 'pred_hoi_logits' in outputs.keys():
            loss_map = {
                'hoi_labels': self.loss_hoi_labels,
                'obj_labels': self.loss_obj_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
            }
        else:
            raise ValueError("outputs must contain \'pred_hoi_logits' for HOI losses")
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['hoi_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))
        if "loss_semantic_consistency" in outputs_without_aux:
            losses["loss_semantic_consistency"] = outputs_without_aux["loss_semantic_consistency"]

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss in {"obj_labels", "hoi_labels"}:
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOITriplet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.subject_category_id = cfg.INPUT.SUBJECT_CATEGORY_ID
        self.num_queries = cfg.MODEL.TRANSFORMER.NUM_QUERIES

    @torch.inference_mode()
    def forward(self, outputs, target_sizes):
        out_hoi_logits = outputs['pred_hoi_logits'].float()
        out_obj_logits = outputs['pred_obj_logits'].float()
        out_sub_boxes = outputs['pred_sub_boxes'].float()
        out_obj_boxes = outputs['pred_obj_boxes'].float()

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            hs, os, ol, sb, ob = hoi_scores[index], obj_scores[index], \
                obj_labels[index], sub_boxes[index], obj_boxes[index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})
            ids = torch.arange(self.num_queries * 2)
            results[-1].update({"hoi_scores": hs.to("cpu"), "obj_scores": os.to("cpu"),
                                "sub_ids": ids[:self.num_queries], "obj_ids": ids[self.num_queries:]})

        return results
