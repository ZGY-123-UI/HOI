import torch
from torch import nn
import torch.nn.functional as F
import warnings

import numpy as np

from models.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from models.detection.layers import build_backbone, TransformerDecoderLayer, TransformerDecoder
from .loss import SetCriterionHOI, PostProcessHOITriplet
from .matcher import build_matcher
from .transformer import build_transformer
from util.misc import NestedTensor, nested_tensor_from_tensor_list


def _cfg_get(node, key, default=None):
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def _cfg_bool(node, key, default=False):
    value = _cfg_get(node, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class RelationPoseSceneAdapter(nn.Module):
    """Build relation-, pose-, and scene-aware visual evidence tokens."""

    def __init__(
        self,
        feature_dim,
        scene_dim,
        num_keypoints=17,
        dim_feedforward=4096,
        dropout=0.1,
        init_scale=0.1,
        eps=1e-6,
    ):
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.eps = float(eps)

        canonical_offsets = torch.tensor(
            [
                [0.00, -0.42],
                [-0.06, -0.45],
                [0.06, -0.45],
                [-0.12, -0.42],
                [0.12, -0.42],
                [-0.22, -0.24],
                [0.22, -0.24],
                [-0.34, -0.04],
                [0.34, -0.04],
                [-0.40, 0.16],
                [0.40, 0.16],
                [-0.16, 0.12],
                [0.16, 0.12],
                [-0.18, 0.34],
                [0.18, 0.34],
                [-0.18, 0.48],
                [0.18, 0.48],
            ],
            dtype=torch.float32,
        )
        if self.num_keypoints > canonical_offsets.shape[0]:
            pad = canonical_offsets.new_zeros(self.num_keypoints - canonical_offsets.shape[0], 2)
            canonical_offsets = torch.cat([canonical_offsets, pad], dim=0)
        self.register_buffer("canonical_offsets", canonical_offsets[: self.num_keypoints])

        bottleneck_dim = min(max(feature_dim // 2, 512), dim_feedforward)
        self.relation_encoder = MLP(12, bottleneck_dim, feature_dim, 3)
        self.pose_encoder = MLP(self.num_keypoints * 6, bottleneck_dim, feature_dim, 3)
        self.scene_encoder = MLP(scene_dim + 4, bottleneck_dim, feature_dim, 3)

        self.relation_norm = nn.LayerNorm(feature_dim)
        self.pose_norm = nn.LayerNorm(feature_dim)
        self.scene_norm = nn.LayerNorm(feature_dim)
        self.context_gate = nn.Linear(feature_dim * 4, 4)
        self.context_fuse = nn.Sequential(
            nn.Linear(feature_dim * 4, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.context_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, pair_feature, human_boxes, object_boxes, token_outputs=None, pose_keypoints=None):
        human_boxes = human_boxes.to(device=pair_feature.device, dtype=pair_feature.dtype)
        object_boxes = object_boxes.to(device=pair_feature.device, dtype=pair_feature.dtype)
        keypoints = self._prepare_keypoints(human_boxes, pose_keypoints)

        relation = self.relation_norm(self.relation_encoder(self._build_spatial_features(human_boxes, object_boxes)))
        pose_inputs = torch.cat(
            [self._build_pose_features(human_boxes, keypoints), self._build_contact_features(object_boxes, keypoints)],
            dim=-1,
        )
        pose = self.pose_norm(self.pose_encoder(pose_inputs))
        scene = self.scene_norm(self.scene_encoder(self._build_scene_features(human_boxes, object_boxes, token_outputs)))

        gate = torch.softmax(self.context_gate(torch.cat([pair_feature, relation, pose, scene], dim=-1)), dim=-1)
        gated = torch.cat(
            [
                pair_feature,
                gate[..., 1:2] * relation,
                gate[..., 2:3] * pose,
                gate[..., 3:4] * scene,
            ],
            dim=-1,
        )
        context = F.normalize(pair_feature + self.context_scale * self.context_fuse(gated), dim=-1)
        return {
            "relation": F.normalize(relation, dim=-1),
            "pose": F.normalize(pose, dim=-1),
            "scene": F.normalize(scene, dim=-1),
            "context": context,
            "gate": gate,
        }

    def _prepare_keypoints(self, human_boxes, pose_keypoints):
        if pose_keypoints is None:
            center = human_boxes[..., :2].unsqueeze(-2)
            size = human_boxes[..., 2:].clamp(min=self.eps).unsqueeze(-2)
            offsets = self.canonical_offsets.to(device=human_boxes.device, dtype=human_boxes.dtype)
            offsets = offsets.view(*([1] * (human_boxes.dim() - 1)), self.num_keypoints, 2)
            xy = center + offsets * size
            conf = torch.ones_like(xy[..., :1])
            return torch.cat([xy, conf], dim=-1)

        keypoints = pose_keypoints.to(device=human_boxes.device, dtype=human_boxes.dtype)
        if keypoints.dim() == human_boxes.dim():
            keypoints = keypoints.unsqueeze(0).expand(human_boxes.shape[0], *keypoints.shape)
        elif keypoints.dim() == human_boxes.dim() + 1 and keypoints.shape[0] == 1 and human_boxes.shape[0] != 1:
            keypoints = keypoints.expand(human_boxes.shape[0], *keypoints.shape[1:])
        if keypoints.shape[-1] == 2:
            keypoints = torch.cat([keypoints, torch.ones_like(keypoints[..., :1])], dim=-1)
        if keypoints.shape[-2] < self.num_keypoints:
            pad_shape = (*keypoints.shape[:-2], self.num_keypoints - keypoints.shape[-2], keypoints.shape[-1])
            keypoints = torch.cat([keypoints, keypoints.new_zeros(pad_shape)], dim=-2)
        return keypoints[..., : self.num_keypoints, :3]

    def _build_pose_features(self, human_boxes, keypoints):
        center = human_boxes[..., :2].unsqueeze(-2)
        size = human_boxes[..., 2:].clamp(min=self.eps).unsqueeze(-2)
        rel_xy = (keypoints[..., :2] - center) / size
        pose = torch.cat([rel_xy, keypoints[..., 2:3].clamp(0.0, 1.0)], dim=-1)
        return pose.flatten(-2)

    def _build_spatial_features(self, human_boxes, object_boxes):
        hcx, hcy, hw, hh = human_boxes.clamp(0.0, 1.0).unbind(-1)
        ocx, ocy, ow, oh = object_boxes.clamp(0.0, 1.0).unbind(-1)
        hw = hw.clamp(min=self.eps)
        hh = hh.clamp(min=self.eps)
        ow = ow.clamp(min=self.eps)
        oh = oh.clamp(min=self.eps)

        raw_dx = ocx - hcx
        raw_dy = ocy - hcy
        dist = torch.sqrt(raw_dx.square() + raw_dy.square() + self.eps)
        human_diag = torch.sqrt(hw.square() + hh.square() + self.eps)
        h_area = (hw * hh).clamp(min=self.eps)
        o_area = (ow * oh).clamp(min=self.eps)

        human_xyxy = self._box_cxcywh_to_xyxy(torch.stack([hcx, hcy, hw, hh], dim=-1))
        object_xyxy = self._box_cxcywh_to_xyxy(torch.stack([ocx, ocy, ow, oh], dim=-1))
        lt = torch.maximum(human_xyxy[..., :2], object_xyxy[..., :2])
        rb = torch.minimum(human_xyxy[..., 2:], object_xyxy[..., 2:])
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]
        union = (h_area + o_area - inter).clamp(min=self.eps)
        iou = inter / union

        return torch.stack(
            [
                raw_dx / hw,
                raw_dy / hh,
                raw_dx / ow,
                raw_dy / oh,
                torch.log(ow / hw),
                torch.log(oh / hh),
                iou,
                o_area / h_area,
                dist / human_diag,
                raw_dx / dist,
                raw_dy / dist,
                union,
            ],
            dim=-1,
        )

    def _build_contact_features(self, object_boxes, keypoints):
        object_xyxy = self._box_cxcywh_to_xyxy(object_boxes.clamp(0.0, 1.0))
        kp_xy = keypoints[..., :2].clamp(0.0, 1.0)
        kp_conf = keypoints[..., 2:3].clamp(0.0, 1.0)

        left_top = object_xyxy[..., :2].unsqueeze(-2)
        right_bottom = object_xyxy[..., 2:].unsqueeze(-2)
        dx = torch.maximum(left_top[..., 0] - kp_xy[..., 0], kp_xy[..., 0] - right_bottom[..., 0]).clamp(min=0.0)
        dy = torch.maximum(left_top[..., 1] - kp_xy[..., 1], kp_xy[..., 1] - right_bottom[..., 1]).clamp(min=0.0)
        obj_size = object_boxes[..., 2:].clamp(min=self.eps).unsqueeze(-2)
        obj_diag = torch.sqrt(obj_size[..., 0].square() + obj_size[..., 1].square() + self.eps)
        dist = torch.sqrt(dx.square() + dy.square() + self.eps).unsqueeze(-1) / obj_diag.unsqueeze(-1)
        inside = ((dx + dy) <= self.eps).to(keypoints.dtype).unsqueeze(-1)
        return torch.cat([dist, inside, kp_conf], dim=-1).flatten(-2)

    def _build_scene_features(self, human_boxes, object_boxes, token_outputs):
        union_xyxy = self._union_xyxy(human_boxes, object_boxes)
        union_cx = (union_xyxy[..., 0] + union_xyxy[..., 2]) * 0.5
        union_cy = (union_xyxy[..., 1] + union_xyxy[..., 3]) * 0.5
        union_w = (union_xyxy[..., 2] - union_xyxy[..., 0]).clamp(min=self.eps)
        union_h = (union_xyxy[..., 3] - union_xyxy[..., 1]).clamp(min=self.eps)
        union_geom = torch.stack([union_cx, union_cy, union_w, union_h], dim=-1)

        if token_outputs is None:
            scene_context = union_geom.new_zeros(*union_geom.shape[:-1], self.scene_encoder.layers[0].in_features - 4)
        else:
            token_outputs = token_outputs.to(device=union_geom.device, dtype=union_geom.dtype)
            scene_context = token_outputs[:, :2].mean(dim=1)
            scene_context = scene_context.unsqueeze(0).unsqueeze(2).expand(*union_geom.shape[:-1], -1)
        return torch.cat([scene_context, union_geom], dim=-1)

    @staticmethod
    def _box_cxcywh_to_xyxy(boxes):
        x_c, y_c, w, h = boxes.unbind(-1)
        return torch.stack(
            [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h],
            dim=-1,
        )

    def _union_xyxy(self, human_boxes, object_boxes):
        human_xyxy = self._box_cxcywh_to_xyxy(human_boxes.clamp(0.0, 1.0))
        object_xyxy = self._box_cxcywh_to_xyxy(object_boxes.clamp(0.0, 1.0))
        lt = torch.minimum(human_xyxy[..., :2], object_xyxy[..., :2])
        rb = torch.maximum(human_xyxy[..., 2:], object_xyxy[..., 2:])
        return torch.cat([lt, rb], dim=-1).clamp(0.0, 1.0)


class MACSemanticCalibrator(nn.Module):
    """Masked semantic recovery and consistency-guided text calibration."""

    def __init__(
        self,
        feature_dim,
        num_slots=4,
        num_heads=8,
        dim_feedforward=4096,
        dropout=0.1,
        temperature=1.0,
        init_scale=0.05,
    ):
        super().__init__()
        self.num_slots = int(num_slots)
        self.temperature = max(float(temperature), 1e-6)
        self.scale = float(feature_dim) ** -0.5

        self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.recovery_delta = nn.Sequential(
            nn.Linear(feature_dim * 3, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.slot_decoder = nn.Sequential(
            nn.Linear(feature_dim * 3, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.num_slots * feature_dim),
        )
        self.calibration_delta = nn.Sequential(
            nn.Linear(feature_dim * 3, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.enh_visual_gate = nn.Linear(feature_dim * 3, feature_dim)
        self.cal_visual_gate = nn.Linear(feature_dim * 4, feature_dim)
        self.enh_text_gate = nn.Linear(feature_dim, feature_dim, bias=False)
        self.cal_text_gate = nn.Linear(feature_dim, feature_dim, bias=False)
        self.recovery_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.calibration_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(
        self,
        pair_feature,
        base_text_embedding,
        masked_text_embedding,
        semantic_prior_embedding,
        visual_evidence,
        logit_scale,
        compute_losses=True,
    ):
        feature_shape = pair_feature.shape
        flat_pair = F.normalize(pair_feature.reshape(-1, feature_shape[-1]), dim=-1)
        base_text = F.normalize(
            base_text_embedding.to(device=pair_feature.device, dtype=pair_feature.dtype), dim=-1
        )
        masked_text = F.normalize(
            masked_text_embedding.to(device=pair_feature.device, dtype=pair_feature.dtype), dim=-1
        )
        prior = self._prepare_prior(
            semantic_prior_embedding, masked_text, pair_feature.device, pair_feature.dtype
        )
        if base_text.shape != masked_text.shape:
            raise ValueError(
                f"MAC-HOI masked prompt bank dim mismatch: base={base_text.shape}, masked={masked_text.shape}"
            )
        if base_text.shape[-1] != flat_pair.shape[-1]:
            raise ValueError(
                f"MAC-HOI feature dim mismatch: pair_feature={pair_feature.shape}, base_text={base_text.shape}"
            )

        evidence = self._flatten_evidence(visual_evidence, pair_feature, flat_pair)
        class_attn = torch.softmax(torch.matmul(flat_pair, base_text.t()) / self.temperature, dim=-1)
        prior_context = torch.einsum("nc,csd->nsd", class_attn, prior)
        masked_context = torch.matmul(class_attn, masked_text).unsqueeze(1)
        attn_kv = torch.cat([prior_context, masked_context, evidence["context"].unsqueeze(1)], dim=1)
        recovered = self.cross_attn(flat_pair.unsqueeze(1), attn_kv, attn_kv, need_weights=False)[0].squeeze(1)

        recovery_input = torch.cat([flat_pair, recovered, evidence["context"]], dim=-1)
        recovery_delta = self.recovery_delta(recovery_input) * self.recovery_scale
        enh_gate = self._class_gate(self.enh_visual_gate(recovery_input), base_text, self.enh_text_gate)
        enhanced_logits = self._score_with_text_delta(flat_pair, base_text, recovery_delta, enh_gate, logit_scale)

        cue_delta = self.calibration_delta(
            torch.cat([evidence["relation"], evidence["pose"], evidence["scene"]], dim=-1)
        ) * self.calibration_scale
        total_delta = recovery_delta + cue_delta
        cal_input = torch.cat([flat_pair, evidence["context"], recovered, total_delta], dim=-1)
        cal_gate = self._class_gate(self.cal_visual_gate(cal_input), base_text, self.cal_text_gate)
        calibrated_logits = self._score_with_text_delta(flat_pair, base_text, total_delta, cal_gate, logit_scale)
        context_logits = self._score_with_text_delta(evidence["context"], base_text, total_delta, cal_gate, logit_scale)

        outputs = {
            "calibrated_logits": calibrated_logits.view(*feature_shape[:-1], -1),
            "enhanced_logits": enhanced_logits.view(*feature_shape[:-1], -1),
            "context_logits": context_logits.view(*feature_shape[:-1], -1),
        }
        losses = {}
        if compute_losses:
            slot_pred = self.slot_decoder(recovery_input).view(-1, self.num_slots, feature_shape[-1])
            slot_target = prior_context[:, : self.num_slots]
            losses["loss_mask_recovery"] = (
                1.0 - F.cosine_similarity(slot_pred, slot_target.detach(), dim=-1)
            ).mean()
            losses["loss_global_proto_consistency"] = self._prototype_consistency_loss(
                base_text, total_delta, cal_gate
            )
        return outputs, losses

    def _prepare_prior(self, semantic_prior_embedding, masked_text, device, dtype):
        if semantic_prior_embedding is None:
            return masked_text.unsqueeze(1).expand(-1, self.num_slots, -1)
        prior = semantic_prior_embedding.to(device=device, dtype=dtype)
        if prior.dim() == 2:
            prior = prior.unsqueeze(1)
        if prior.shape[1] < self.num_slots:
            pad = prior[:, -1:].expand(-1, self.num_slots - prior.shape[1], -1)
            prior = torch.cat([prior, pad], dim=1)
        return F.normalize(prior[:, : self.num_slots], dim=-1)

    def _flatten_evidence(self, visual_evidence, pair_feature, flat_pair):
        if visual_evidence is None:
            zero = torch.zeros_like(flat_pair)
            return {"relation": zero, "pose": zero, "scene": zero, "context": flat_pair}
        evidence = {}
        for key in ["relation", "pose", "scene", "context"]:
            value = visual_evidence.get(key)
            if value is None:
                value = torch.zeros_like(pair_feature) if key != "context" else pair_feature
            evidence[key] = F.normalize(value.reshape(-1, pair_feature.shape[-1]), dim=-1)
        return evidence

    def _class_gate(self, visual_gate_feature, base_text, text_gate):
        text_feature = text_gate(base_text)
        return torch.sigmoid(torch.matmul(visual_gate_feature, text_feature.t()) * self.scale)

    def _score_with_text_delta(self, visual, base_text, delta, gate, logit_scale):
        visual = F.normalize(visual, dim=-1)
        base_text = F.normalize(base_text, dim=-1)
        base_logits = torch.matmul(visual, base_text.t())
        visual_delta = (visual * delta).sum(dim=-1, keepdim=True)
        base_delta = torch.matmul(delta, base_text.t())
        delta_norm_sq = delta.square().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        text_norm = torch.sqrt((1.0 + 2.0 * gate * base_delta + gate.square() * delta_norm_sq).clamp(min=1e-6))
        return logit_scale * (base_logits + gate * visual_delta) / text_norm

    def _prototype_consistency_loss(self, base_text, delta, gate):
        if delta.numel() == 0:
            return base_text.sum() * 0.0
        avg_delta = delta.mean(dim=0, keepdim=True)
        avg_gate = gate.mean(dim=0).unsqueeze(-1)
        calibrated_text = F.normalize(base_text + avg_gate * avg_delta, dim=-1)
        base_sim = torch.matmul(base_text, base_text.t())
        calibrated_sim = torch.matmul(calibrated_text, calibrated_text.t())
        return F.mse_loss(calibrated_sim, base_sim)


class PoseSpatialInteractionEncoder(nn.Module):
    """Encode body-part layout, human-object geometry, and contact cues together."""

    def __init__(
        self,
        hidden_dim,
        dino_embed_dim,
        num_keypoints=6,
        dropout=0.1,
        init_scale=0.1,
        eps=1e-6,
    ):
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.eps = float(eps)

        # COCO-style body anchors in normalized human-box offsets:
        # nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
        canonical_offsets = torch.tensor(
            [
                [0.00, -0.42],
                [-0.06, -0.45],
                [0.06, -0.45],
                [-0.12, -0.42],
                [0.12, -0.42],
                [-0.22, -0.24],
                [0.22, -0.24],
                [-0.34, -0.04],
                [0.34, -0.04],
                [-0.40, 0.16],
                [0.40, 0.16],
                [-0.16, 0.12],
                [0.16, 0.12],
                [-0.18, 0.34],
                [0.18, 0.34],
                [-0.18, 0.48],
                [0.18, 0.48],
            ],
            dtype=torch.float32,
        )
        if self.num_keypoints > canonical_offsets.shape[0]:
            pad = canonical_offsets.new_zeros(self.num_keypoints - canonical_offsets.shape[0], 2)
            canonical_offsets = torch.cat([canonical_offsets, pad], dim=0)
        self.register_buffer("canonical_offsets", canonical_offsets[: self.num_keypoints])

        bottleneck_dim = max(hidden_dim, min(dino_embed_dim, hidden_dim * 2))
        self.pose_encoder = MLP(self.num_keypoints * 3, bottleneck_dim, bottleneck_dim, 3)
        self.spatial_encoder = MLP(12, bottleneck_dim, bottleneck_dim, 3)
        self.contact_encoder = MLP(self.num_keypoints * 3, bottleneck_dim, bottleneck_dim, 3)
        self.fusion = nn.Sequential(
            nn.Linear(bottleneck_dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(bottleneck_dim),
        )
        self.to_hidden = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.to_dino = nn.Sequential(
            nn.Linear(bottleneck_dim, dino_embed_dim),
            nn.LayerNorm(dino_embed_dim),
        )
        self.hidden_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.dino_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, human_boxes, object_boxes, pose_keypoints=None):
        keypoints = self._prepare_keypoints(human_boxes, pose_keypoints)
        pose_feature = self._build_pose_features(human_boxes, keypoints)
        spatial_feature = self._build_spatial_features(human_boxes, object_boxes)
        contact_feature = self._build_contact_features(object_boxes, keypoints)

        fused = self.fusion(torch.cat([
            self.pose_encoder(pose_feature),
            self.spatial_encoder(spatial_feature),
            self.contact_encoder(contact_feature),
        ], dim=-1))
        hidden_delta = self.to_hidden(fused) * self.hidden_scale
        dino_delta = self.to_dino(fused) * self.dino_scale
        return hidden_delta, dino_delta

    def _prepare_keypoints(self, human_boxes, pose_keypoints):
        if pose_keypoints is None:
            center = human_boxes[..., :2].unsqueeze(-2)
            size = human_boxes[..., 2:].clamp(min=self.eps).unsqueeze(-2)
            offsets = self.canonical_offsets.to(device=human_boxes.device, dtype=human_boxes.dtype)
            offsets = offsets.view(*([1] * (human_boxes.dim() - 1)), self.num_keypoints, 2)
            xy = center + offsets * size
            conf = torch.ones_like(xy[..., :1])
            return torch.cat([xy, conf], dim=-1)

        keypoints = pose_keypoints.to(device=human_boxes.device, dtype=human_boxes.dtype)
        if keypoints.dim() == human_boxes.dim():
            keypoints = keypoints.unsqueeze(0).expand(human_boxes.shape[0], *keypoints.shape)
        elif keypoints.dim() == human_boxes.dim() + 1 and keypoints.shape[0] == 1 and human_boxes.shape[0] != 1:
            keypoints = keypoints.expand(human_boxes.shape[0], *keypoints.shape[1:])
        if keypoints.shape[-2] < self.num_keypoints:
            pad_shape = (*keypoints.shape[:-2], self.num_keypoints - keypoints.shape[-2], keypoints.shape[-1])
            keypoints = torch.cat([keypoints, keypoints.new_zeros(pad_shape)], dim=-2)
        keypoints = keypoints[..., : self.num_keypoints, :3]
        if keypoints.shape[-1] == 2:
            conf = torch.ones_like(keypoints[..., :1])
            keypoints = torch.cat([keypoints, conf], dim=-1)
        return keypoints

    def _build_pose_features(self, human_boxes, keypoints):
        center = human_boxes[..., :2].unsqueeze(-2)
        size = human_boxes[..., 2:].clamp(min=self.eps).unsqueeze(-2)
        rel_xy = (keypoints[..., :2] - center) / size
        pose = torch.cat([rel_xy, keypoints[..., 2:3].clamp(0.0, 1.0)], dim=-1)
        return pose.flatten(-2)

    def _build_spatial_features(self, human_boxes, object_boxes):
        hcx, hcy, hw, hh = human_boxes.clamp(0.0, 1.0).unbind(-1)
        ocx, ocy, ow, oh = object_boxes.clamp(0.0, 1.0).unbind(-1)
        hw = hw.clamp(min=self.eps)
        hh = hh.clamp(min=self.eps)
        ow = ow.clamp(min=self.eps)
        oh = oh.clamp(min=self.eps)

        raw_dx = ocx - hcx
        raw_dy = ocy - hcy
        dist = torch.sqrt(raw_dx.square() + raw_dy.square() + self.eps)
        human_diag = torch.sqrt(hw.square() + hh.square() + self.eps)
        h_area = (hw * hh).clamp(min=self.eps)
        o_area = (ow * oh).clamp(min=self.eps)

        human_xyxy = self._box_cxcywh_to_xyxy(torch.stack([hcx, hcy, hw, hh], dim=-1))
        object_xyxy = self._box_cxcywh_to_xyxy(torch.stack([ocx, ocy, ow, oh], dim=-1))
        lt = torch.maximum(human_xyxy[..., :2], object_xyxy[..., :2])
        rb = torch.minimum(human_xyxy[..., 2:], object_xyxy[..., 2:])
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]
        union = (h_area + o_area - inter).clamp(min=self.eps)
        iou = inter / union

        return torch.stack(
            [
                raw_dx / hw,
                raw_dy / hh,
                raw_dx / ow,
                raw_dy / oh,
                torch.log(ow / hw),
                torch.log(oh / hh),
                iou,
                o_area / h_area,
                dist / human_diag,
                raw_dx / dist,
                raw_dy / dist,
                union,
            ],
            dim=-1,
        )

    def _build_contact_features(self, object_boxes, keypoints):
        object_xyxy = self._box_cxcywh_to_xyxy(object_boxes.clamp(0.0, 1.0))
        kp_xy = keypoints[..., :2].clamp(0.0, 1.0)
        kp_conf = keypoints[..., 2:3].clamp(0.0, 1.0)

        left_top = object_xyxy[..., :2].unsqueeze(-2)
        right_bottom = object_xyxy[..., 2:].unsqueeze(-2)
        dx = torch.maximum(left_top[..., 0] - kp_xy[..., 0], kp_xy[..., 0] - right_bottom[..., 0]).clamp(min=0.0)
        dy = torch.maximum(left_top[..., 1] - kp_xy[..., 1], kp_xy[..., 1] - right_bottom[..., 1]).clamp(min=0.0)
        obj_size = object_boxes[..., 2:].clamp(min=self.eps).unsqueeze(-2)
        obj_diag = torch.sqrt(obj_size[..., 0].square() + obj_size[..., 1].square() + self.eps)
        dist = torch.sqrt(dx.square() + dy.square() + self.eps).unsqueeze(-1) / obj_diag.unsqueeze(-1)
        inside = ((dx + dy) <= self.eps).to(keypoints.dtype).unsqueeze(-1)
        return torch.cat([dist, inside, kp_conf], dim=-1).flatten(-2)

    @staticmethod
    def _box_cxcywh_to_xyxy(boxes):
        x_c, y_c, w, h = boxes.unbind(-1)
        return torch.stack(
            [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h],
            dim=-1,
        )


class ViTPoseTopDownEstimator(nn.Module):
    """Run an external ViTPose/MMPose top-down model on predicted human boxes."""

    def __init__(
        self,
        config,
        checkpoint,
        num_keypoints,
        score_thr=0.05,
        image_format="RGB",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.config = config
        self.checkpoint = checkpoint
        self.num_keypoints = int(num_keypoints)
        self.score_thr = float(score_thr)
        self.image_format = str(image_format).upper()
        self._model_holder = [None]
        self._device_holder = [None]
        self.register_buffer("pixel_mean", torch.tensor(mean).view(3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(std).view(3, 1, 1), persistent=False)

    def forward(self, samples, human_boxes):
        if human_boxes.dim() == 4:
            num_layers, batch_size, num_queries, _ = human_boxes.shape
            boxes_for_pose = human_boxes[-1]
        else:
            num_layers = None
            batch_size, num_queries, _ = human_boxes.shape
            boxes_for_pose = human_boxes

        device = boxes_for_pose.device
        dtype = boxes_for_pose.dtype
        pose_keypoints = boxes_for_pose.new_zeros(batch_size, num_queries, self.num_keypoints, 3)
        pose_model = self._get_model(str(device))

        tensors = samples.tensors.detach()
        mask = samples.mask
        for batch_idx in range(batch_size):
            valid_h, valid_w = self._valid_image_size(mask[batch_idx], tensors.shape[-2], tensors.shape[-1])
            image = self._restore_image(tensors[batch_idx], valid_h, valid_w)
            boxes_xyxy = self._boxes_to_pixel_xyxy(boxes_for_pose[batch_idx], valid_h, valid_w)
            boxes_np = boxes_xyxy.detach().float().cpu().numpy()
            keep = self._valid_boxes(boxes_np)
            if not keep.any():
                continue

            predictions = self._inference_topdown(pose_model, image, boxes_np[keep])
            query_indices = torch.as_tensor(np.nonzero(keep)[0], device=device, dtype=torch.long)
            parsed = self._parse_predictions(predictions, len(query_indices), valid_h, valid_w, device, dtype)
            pose_keypoints[batch_idx, query_indices] = parsed

        if num_layers is not None:
            pose_keypoints = pose_keypoints.unsqueeze(0).expand(num_layers, -1, -1, -1, -1)
        return pose_keypoints

    def _get_model(self, device):
        if not self.config or not self.checkpoint:
            raise ValueError(
                "MODEL.VITPOSE.CONFIG and MODEL.VITPOSE.CHECKPOINT must be set when ViTPose is enabled."
            )
        if self._model_holder[0] is not None and self._device_holder[0] == device:
            return self._model_holder[0]
        try:
            from mmpose.apis import init_model
            from mmpose.utils import register_all_modules
        except ImportError as exc:
            raise ImportError(
                "ViTPose integration requires MMPose. Install mmpose/mmcv/mmengine "
                "and set MODEL.VITPOSE.CONFIG / MODEL.VITPOSE.CHECKPOINT."
            ) from exc
        register_all_modules(init_default_scope=True)
        model = init_model(self.config, self.checkpoint, device=device)
        model.eval()
        self._model_holder[0] = model
        self._device_holder[0] = device
        return model

    def _inference_topdown(self, model, image, boxes_xyxy):
        try:
            from mmpose.apis import inference_topdown
        except ImportError as exc:
            raise ImportError("MMPose inference_topdown API is required for ViTPose inference.") from exc
        return inference_topdown(model, image, boxes_xyxy)

    def _restore_image(self, tensor, valid_h, valid_w):
        image = tensor.detach().float().cpu()
        mean = self.pixel_mean.detach().float().cpu()
        std = self.pixel_std.detach().float().cpu()
        image = (image * std + mean).clamp(0.0, 1.0)
        image = image[:, :valid_h, :valid_w].permute(1, 2, 0).numpy()
        image = (image * 255.0).round().astype(np.uint8)
        if self.image_format == "BGR":
            image = image[..., ::-1]
        return image

    @staticmethod
    def _valid_image_size(mask, padded_h, padded_w):
        if mask is None:
            return int(padded_h), int(padded_w)
        valid = ~mask
        if valid.any():
            ys, xs = valid.nonzero(as_tuple=True)
            return int(ys.max().item() + 1), int(xs.max().item() + 1)
        return int(padded_h), int(padded_w)

    @staticmethod
    def _boxes_to_pixel_xyxy(boxes, height, width):
        cx, cy, w, h = boxes.clamp(0.0, 1.0).unbind(-1)
        x1 = (cx - 0.5 * w) * width
        y1 = (cy - 0.5 * h) * height
        x2 = (cx + 0.5 * w) * width
        y2 = (cy + 0.5 * h) * height
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        boxes_xyxy[..., 0::2] = boxes_xyxy[..., 0::2].clamp(0.0, float(width - 1))
        boxes_xyxy[..., 1::2] = boxes_xyxy[..., 1::2].clamp(0.0, float(height - 1))
        return boxes_xyxy

    def _valid_boxes(self, boxes_xyxy):
        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        return (widths > 1.0) & (heights > 1.0)

    def _parse_predictions(self, predictions, expected, height, width, device, dtype):
        keypoints = torch.zeros(expected, self.num_keypoints, 3, device=device, dtype=dtype)
        for idx, pred in enumerate(predictions[:expected]):
            xy, scores = self._extract_keypoints(pred)
            if xy is None:
                continue
            xy = xy[: self.num_keypoints]
            scores = scores[: self.num_keypoints] if scores is not None else np.ones((xy.shape[0],), dtype=np.float32)
            if xy.shape[0] < self.num_keypoints:
                pad_xy = np.zeros((self.num_keypoints - xy.shape[0], 2), dtype=np.float32)
                pad_scores = np.zeros((self.num_keypoints - xy.shape[0],), dtype=np.float32)
                xy = np.concatenate([xy, pad_xy], axis=0)
                scores = np.concatenate([scores, pad_scores], axis=0)
            xy_norm = xy.astype(np.float32).copy()
            xy_norm[:, 0] = xy_norm[:, 0] / max(float(width), 1.0)
            xy_norm[:, 1] = xy_norm[:, 1] / max(float(height), 1.0)
            scores = scores.astype(np.float32)
            scores = np.where(scores >= self.score_thr, scores, 0.0)
            pose = np.concatenate([xy_norm, scores[:, None]], axis=-1)
            keypoints[idx] = torch.as_tensor(pose, device=device, dtype=dtype)
        return keypoints

    @staticmethod
    def _extract_keypoints(prediction):
        if hasattr(prediction, "pred_instances"):
            instances = prediction.pred_instances
            keypoints = getattr(instances, "keypoints", None)
            scores = getattr(instances, "keypoint_scores", None)
            if keypoints is not None:
                keypoints = np.asarray(keypoints)
                if keypoints.ndim == 3:
                    keypoints = keypoints[0]
                if scores is not None:
                    scores = np.asarray(scores)
                    if scores.ndim == 2:
                        scores = scores[0]
                return keypoints, scores
        if isinstance(prediction, dict):
            keypoints = prediction.get("keypoints", None)
            scores = prediction.get("keypoint_scores", prediction.get("keypoint_scores_pred", None))
            if keypoints is not None:
                keypoints = np.asarray(keypoints)
                if keypoints.ndim == 3:
                    keypoints = keypoints[0]
                if scores is not None:
                    scores = np.asarray(scores)
                    if scores.ndim == 2:
                        scores = scores[0]
                elif keypoints.shape[-1] >= 3:
                    scores = keypoints[:, 2]
                    keypoints = keypoints[:, :2]
                return keypoints, scores
        return None, None


class HOIDetector(nn.Module):
    def __init__(self, cfg, backbone, head_model, transformer):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

        self.num_queries = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.backbone = backbone
        self.head_model = head_model
        self.num_register_tokens = cfg.MODEL.DINO.N_STORAGE_TOKENS
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed_h = nn.Embedding(cfg.MODEL.TRANSFORMER.NUM_QUERIES, hidden_dim)
        self.query_embed_o = nn.Embedding(cfg.MODEL.TRANSFORMER.NUM_QUERIES, hidden_dim)
        self.pos_guided_embed = nn.Embedding(cfg.MODEL.TRANSFORMER.NUM_QUERIES, hidden_dim)

        self.input_proj = nn.Sequential(nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                                        nn.GroupNorm(32, hidden_dim),)

        dino_embed_dim = cfg.MODEL.DINO.EMBED_DIM
        self.semantic_proj = nn.Sequential(
            nn.Linear(dino_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, dino_embed_dim),
            nn.LayerNorm(dino_embed_dim)
        )
        fusion_decoder_layer = TransformerDecoderLayer(
            dino_embed_dim, 16, dino_embed_dim * 4,
            dropout=0.1, activation="relu"
        )
        num_fus_dec_layers = 1
        fusion_decoder_norm = nn.LayerNorm(dino_embed_dim)
        self.fusion_decoder = TransformerDecoder(
            fusion_decoder_layer,
            num_fus_dec_layers,
            fusion_decoder_norm,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.hoi_class_fc = nn.Sequential(
            nn.Linear(dino_embed_dim, dino_embed_dim * 2),
            nn.LayerNorm(dino_embed_dim * 2)
        )
        self.num_dec_layers = cfg.MODEL.TRANSFORMER.INS_DEC_LAYERS

        self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self.dino_embed_dim = cfg.MODEL.DINO.EMBED_DIM
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        num_obj_classes = cfg.INPUT.NUM_OBJ_CLASSES

        classifier_train_weights = torch.load(cfg.ZERO_SHOT.CLASSIFIER.TRAIN, map_location="cpu", weights_only=True)
        classifier_eval_weights = torch.load(cfg.ZERO_SHOT.CLASSIFIER.EVAL, map_location="cpu", weights_only=True)

        if cfg.MODEL.DINO.WITH_DINO_LABEL:
            self.hoi_class_fc = nn.Sequential(
                nn.Linear(cfg.MODEL.DINO.EMBED_DIM, 2*cfg.MODEL.DINO.EMBED_DIM),
                nn.LayerNorm(2*cfg.MODEL.DINO.EMBED_DIM),
            )
            hoi_embedding_train = classifier_train_weights["hoi_embedding_train"]
            self.visual_projection = nn.Linear(2*cfg.MODEL.DINO.EMBED_DIM, hoi_embedding_train.shape[0], bias=False)
            with torch.no_grad():
                self.visual_projection.weight.copy_(hoi_embedding_train)
            if cfg.INPUT.DATASET_FILE == "hico" and cfg.ZERO_SHOT.TYPE != "default":
                hoi_embedding_eval = classifier_eval_weights["hoi_embedding_eval"]
                self.eval_visual_projection = nn.Linear(2*cfg.MODEL.DINO.EMBED_DIM, hoi_embedding_eval.shape[0], bias=False)
                with torch.no_grad():
                    self.eval_visual_projection.weight.copy_(hoi_embedding_eval)
        else:
            raise ValueError("Please use DINO.txt label for HOI classification!")

        if cfg.MODEL.DINO.WITH_OBJ_DINO_LABEL:
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, 2*cfg.MODEL.DINO.EMBED_DIM),
                nn.LayerNorm(2*cfg.MODEL.DINO.EMBED_DIM),
            )
            obj_embedding = classifier_eval_weights["obj_text_embedding_eval"]
            self.obj_visual_projection = nn.Linear(2*cfg.MODEL.DINO.EMBED_DIM, num_obj_classes + 1)
            with torch.no_grad():
                self.obj_visual_projection.weight.copy_(obj_embedding)
        else:
            raise ValueError("Please use DINO.txt label for object classification!")

        sem_cfg = _cfg_get(cfg.MODEL, "SEMANTIC_ENHANCEMENT", {})
        self.use_semantic_enhancement = _cfg_bool(sem_cfg, "ENABLED")
        if self.use_semantic_enhancement:
            require_masked_embedding = _cfg_bool(sem_cfg, "REQUIRE_MASKED_EMBEDDING", False)
            if "hoi_embedding_masked_train" not in classifier_train_weights:
                if require_masked_embedding:
                    raise KeyError(
                        "MODEL.SEMANTIC_ENHANCEMENT requires 'hoi_embedding_masked_train'. "
                        "Please rebuild HICO classifier weights with hico_offline_classifier.py."
                    )
                warnings.warn(
                    "HICO masked prompt embedding not found in classifier weights; "
                    "falling back to full HOI prompt embeddings.",
                    RuntimeWarning,
                )
            masked_train_embedding = classifier_train_weights.get("hoi_embedding_masked_train", hoi_embedding_train)
            self.register_buffer("hoi_masked_embedding_train", masked_train_embedding.float())
            masked_eval_embedding = classifier_eval_weights.get(
                "hoi_embedding_masked_eval",
                classifier_eval_weights.get("hoi_embedding_eval", masked_train_embedding),
            )
            self.register_buffer("hoi_masked_embedding_eval", masked_eval_embedding.float())
            prior_train_embedding = classifier_train_weights.get(
                "hoi_semantic_prior_train",
                classifier_train_weights.get("hoi_masked_slot_embedding_train", None),
            )
            if prior_train_embedding is None:
                prior_train_embedding = masked_train_embedding.float().unsqueeze(1).expand(-1, 4, -1).clone()
            prior_eval_embedding = classifier_eval_weights.get(
                "hoi_semantic_prior_eval",
                classifier_eval_weights.get("hoi_masked_slot_embedding_eval", None),
            )
            if prior_eval_embedding is None:
                prior_eval_embedding = masked_eval_embedding.float().unsqueeze(1).expand(-1, 4, -1).clone()
            self.register_buffer("hoi_semantic_prior_train", prior_train_embedding.float())
            self.register_buffer("hoi_semantic_prior_eval", prior_eval_embedding.float())
            self.semantic_calibrator = MACSemanticCalibrator(
                feature_dim=2 * cfg.MODEL.DINO.EMBED_DIM,
                num_slots=int(_cfg_get(sem_cfg, "NUM_SLOTS", 4)),
                num_heads=int(_cfg_get(sem_cfg, "NUM_HEADS", 8)),
                dim_feedforward=int(_cfg_get(sem_cfg, "DIM_FEEDFORWARD", 4096)),
                dropout=float(_cfg_get(sem_cfg, "DROPOUT", 0.1)),
                temperature=float(_cfg_get(sem_cfg, "TEMPERATURE", 1.0)),
                init_scale=float(_cfg_get(sem_cfg, "INIT_SCALE", 0.05)),
            )
        else:
            self.semantic_calibrator = None

        rps_cfg = _cfg_get(
            cfg.MODEL,
            "RELATION_POSE_SCENE_ADAPTER",
            _cfg_get(cfg.MODEL, "POSE_SPATIAL_INTERACTION", {}),
        )
        self.use_rps_adapter = _cfg_bool(rps_cfg, "ENABLED", False)
        self.rps_detach_boxes = _cfg_bool(rps_cfg, "DETACH_BOXES", True)
        self.rps_num_keypoints = int(_cfg_get(rps_cfg, "NUM_KEYPOINTS", 17))
        if self.use_rps_adapter:
            self.rps_adapter = RelationPoseSceneAdapter(
                feature_dim=2 * cfg.MODEL.DINO.EMBED_DIM,
                scene_dim=dino_embed_dim,
                num_keypoints=self.rps_num_keypoints,
                dim_feedforward=int(_cfg_get(rps_cfg, "DIM_FEEDFORWARD", 4096)),
                dropout=float(_cfg_get(rps_cfg, "DROPOUT", 0.1)),
                init_scale=float(_cfg_get(rps_cfg, "INIT_SCALE", 0.1)),
            )
        else:
            self.rps_adapter = None
        self.pose_spatial_num_keypoints = self.rps_num_keypoints
        self.pose_spatial_encoder = None

        ps_cfg = _cfg_get(cfg.MODEL, "POSE_SPATIAL_INTERACTION", {})
        self.use_pose_spatial_interaction = _cfg_bool(ps_cfg, "ENABLED", False)
        if self.use_pose_spatial_interaction and self.rps_adapter is None:
            warnings.warn(
                "MODEL.POSE_SPATIAL_INTERACTION is deprecated; use "
                "MODEL.RELATION_POSE_SCENE_ADAPTER for MAC-HOI.",
                RuntimeWarning,
            )
            self.pose_spatial_encoder = PoseSpatialInteractionEncoder(
                hidden_dim=hidden_dim,
                dino_embed_dim=dino_embed_dim,
                num_keypoints=self.pose_spatial_num_keypoints,
                dropout=float(_cfg_get(ps_cfg, "DROPOUT", 0.1)),
                init_scale=float(_cfg_get(ps_cfg, "INIT_SCALE", 0.1)),
            )
            self.pose_spatial_detach_boxes = _cfg_bool(ps_cfg, "DETACH_BOXES", True)
        else:
            self.pose_spatial_detach_boxes = True

        vitpose_cfg = _cfg_get(cfg.MODEL, "VITPOSE", {})
        self.use_vitpose = _cfg_bool(vitpose_cfg, "ENABLED", False)
        if self.use_vitpose:
            self.vitpose_estimator = ViTPoseTopDownEstimator(
                config=_cfg_get(vitpose_cfg, "CONFIG", ""),
                checkpoint=_cfg_get(vitpose_cfg, "CHECKPOINT", ""),
                num_keypoints=int(_cfg_get(vitpose_cfg, "NUM_KEYPOINTS", self.pose_spatial_num_keypoints)),
                score_thr=float(_cfg_get(vitpose_cfg, "SCORE_THR", 0.05)),
                image_format=_cfg_get(vitpose_cfg, "IMAGE_FORMAT", "RGB"),
                mean=tuple(_cfg_get(vitpose_cfg, "PIXEL_MEAN", (0.485, 0.456, 0.406))),
                std=tuple(_cfg_get(vitpose_cfg, "PIXEL_STD", (0.229, 0.224, 0.225))),
            )
        else:
            self.vitpose_estimator = None

        self.aux_loss = cfg.MODEL.HOI.DEEP_SUPERVISION
        self.hidden_dim = hidden_dim
        self.cfg = cfg
        self.reset_parameters()
        self.freeze()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.cfg.MODEL.DINO.FREEZE:
                self.backbone.eval()
        return self

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head_model.parameters():
            p.requires_grad = False
        for p in self.visual_projection.parameters():
            p.requires_grad = False
        for p in self.obj_visual_projection.parameters():
            p.requires_grad = False
        if self.cfg.ZERO_SHOT.TYPE != "default":
            for p in self.eval_visual_projection.parameters():
                p.requires_grad = False

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embed.weight)

    @torch.inference_mode()
    def get_semantics(self, tokens):
        image_tokens = self.head_model(tokens)
        class_token = image_tokens[:, 0]
        patch_tokens = image_tokens[:, self.num_register_tokens + 1 :]
        semantics = torch.cat([
            class_token.unsqueeze(1),
            torch.mean(patch_tokens, dim=1, keepdim=True),
            patch_tokens
        ], dim=1)
        return semantics

    def get_query_outputs(self, tokens, inter_query, return_token_outputs=False):
        bs, num_queries, embed_dim = inter_query.shape
        assert num_queries == self.num_queries, f"num_queries should be {self.num_queries}, but got {num_queries}"

        class_and_register_tokens = tokens[:, :self.num_register_tokens + 1, :]
        patch_tokens = tokens[:, self.num_register_tokens + 1:, :]
        query_tokens = torch.cat([
            class_and_register_tokens,
            inter_query,
            patch_tokens
        ], dim=1)

        query_start = self.num_register_tokens + 1
        query_end = query_start + num_queries

        token_outputs = self.head_model(
            query_tokens,
        )
        query_tokens = token_outputs[:, query_start:query_end, :]
        if return_token_outputs:
            class_token, patch_tokens = token_outputs[:, 0], token_outputs[:, query_end:]
            token_outputs = torch.cat((
                class_token.unsqueeze(1), 
                torch.mean(patch_tokens, dim=1, keepdim=True), 
                patch_tokens), dim=1
            )
            return query_tokens, token_outputs
        else:
            return query_tokens

    @torch.inference_mode()
    def get_backbone_features(self, samples):
        features, pos, tokens = self.backbone(samples, return_tokens=True)
        src, mask = features[-1].decompose()
        assert mask is not None
        return src, mask, pos[-1], tokens
    
    def forward(self, samples: NestedTensor, pose_keypoints=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        src, mask, pos, tokens = self.get_backbone_features(samples)
        # for object detection, we do not need gradients for semantics
        semantics = self.get_semantics(tokens)

        if self.training:
            src = src.clone().detach().requires_grad_(True)
            pos = pos.clone().detach().requires_grad_(True)
            tokens = tokens.clone().detach().requires_grad_(True)
            semantics = semantics.clone().detach().requires_grad_(True)

        h_hs, o_hs = self.transformer(
            self.input_proj(src), mask,
            self.query_embed_h.weight,
            self.query_embed_o.weight,
            self.pos_guided_embed.weight,
            pos, self.semantic_proj(semantics)
        )

        outputs_sub_coord = self.hum_bbox_embed(h_hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(o_hs).sigmoid()

        needs_pose_keypoints = self.pose_spatial_encoder is not None or self.rps_adapter is not None
        if pose_keypoints is None and self.vitpose_estimator is not None and needs_pose_keypoints:
            pose_keypoints = self.vitpose_estimator(samples, outputs_sub_coord.detach())

        pose_spatial_hidden = None
        pose_spatial_dino = None
        if self.pose_spatial_encoder is not None:
            ps_sub_boxes = outputs_sub_coord.detach() if self.pose_spatial_detach_boxes else outputs_sub_coord
            ps_obj_boxes = outputs_obj_coord.detach() if self.pose_spatial_detach_boxes else outputs_obj_coord
            pose_spatial_hidden, pose_spatial_dino = self.pose_spatial_encoder(
                ps_sub_boxes, ps_obj_boxes, pose_keypoints
            )

        inter_query = (h_hs + o_hs) / 2.0
        if pose_spatial_hidden is not None:
            inter_query = inter_query + pose_spatial_hidden.to(dtype=inter_query.dtype)

        inter_hs_list = []
        for i in range(self.num_dec_layers):
            if i == self.num_dec_layers - 1: # return token_outputs only for the last layer
                query_tokens, token_outputs = self.get_query_outputs(
                    tokens, 
                    self.query_proj(inter_query[i]),
                    return_token_outputs=True
                )
            else:
                query_tokens = self.get_query_outputs(
                    tokens, 
                    self.query_proj(inter_query[i])
                )
            inter_hs_list.append(query_tokens)
        inter_hs = torch.stack(inter_hs_list, dim=0)

        # fusion inter_hs with token_outputs via fusion_decoder
        fused_inter_hs_list = []
        for i in range(self.num_dec_layers):
            fused_inter_hs_list.append(
                self.fusion_decoder(inter_hs[i].transpose(0, 1), token_outputs.transpose(0, 1))
            )
        inter_hs = torch.stack(fused_inter_hs_list, dim=0).transpose(1, 2)
        if pose_spatial_dino is not None:
            inter_hs = inter_hs + pose_spatial_dino.to(dtype=inter_hs.dtype)

        if self.cfg.MODEL.DINO.WITH_OBJ_DINO_LABEL:
            obj_logit_scale = self.obj_logit_scale.exp()
            o_hs = self.obj_class_fc(o_hs)
            o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
            outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)
        else:
            raise ValueError("Please use DINO.txt label for object classification!")

        if self.cfg.MODEL.DINO.WITH_DINO_LABEL:
            logit_scale = self.logit_scale.exp()
            inter_hs = self.hoi_class_fc(inter_hs)
            inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
            visual_evidence = None
            if self.rps_adapter is not None:
                rps_sub_boxes = outputs_sub_coord.detach() if self.rps_detach_boxes else outputs_sub_coord
                rps_obj_boxes = outputs_obj_coord.detach() if self.rps_detach_boxes else outputs_obj_coord
                visual_evidence = self.rps_adapter(
                    inter_hs,
                    rps_sub_boxes,
                    rps_obj_boxes,
                    token_outputs=token_outputs,
                    pose_keypoints=pose_keypoints,
                )
            use_eval_hoi_bank = (
                self.cfg.INPUT.DATASET_FILE == "hico"
                and self.cfg.ZERO_SHOT.TYPE != "default"
                and (self.cfg.RUNTIME.EVAL or not self.training)
            )
            if use_eval_hoi_bank:
                base_hoi_embedding = self.eval_visual_projection.weight
                outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
                masked_hoi_embedding = self.hoi_masked_embedding_eval if self.semantic_calibrator is not None else None
                semantic_prior_embedding = self.hoi_semantic_prior_eval if self.semantic_calibrator is not None else None
            else:
                base_hoi_embedding = self.visual_projection.weight
                outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)
                masked_hoi_embedding = self.hoi_masked_embedding_train if self.semantic_calibrator is not None else None
                semantic_prior_embedding = self.hoi_semantic_prior_train if self.semantic_calibrator is not None else None

            semantic_losses = {}
            outputs_hoi_masked_class = None
            outputs_ctx_hoi_class = None
            if self.semantic_calibrator is not None:
                calibrated_outputs, semantic_losses = self.semantic_calibrator(
                    inter_hs,
                    base_hoi_embedding,
                    masked_hoi_embedding,
                    semantic_prior_embedding,
                    visual_evidence,
                    logit_scale,
                    compute_losses=self.training,
                )
                outputs_hoi_class = calibrated_outputs["calibrated_logits"]
                outputs_hoi_masked_class = calibrated_outputs["enhanced_logits"]
                outputs_ctx_hoi_class = calibrated_outputs["context_logits"]
        else:
            raise ValueError("Please use DINO.txt label for HOI classification!")

        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_obj_logits': outputs_obj_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        for loss_name, loss_value in semantic_losses.items():
            out[loss_name] = loss_value
        if outputs_hoi_masked_class is not None:
            out['pred_hoi_masked_logits'] = outputs_hoi_masked_class[-1]
        if outputs_ctx_hoi_class is not None:
            out['pred_ctx_hoi_logits'] = outputs_ctx_hoi_class[-1]

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss_triplet(outputs_hoi_class, outputs_obj_class,
                                                            outputs_sub_coord, outputs_obj_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_hoi_class, outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord):

        num_dec_layers = self.num_dec_layers
        aux_outputs = {'pred_hoi_logits': outputs_hoi_class[-num_dec_layers: -1],
                       'pred_obj_logits': outputs_obj_class[-num_dec_layers: -1],
                       'pred_sub_boxes': outputs_sub_coord[-num_dec_layers: -1],
                       'pred_obj_boxes': outputs_obj_coord[-num_dec_layers: -1]}
        outputs_auxes = []
        for i in range(num_dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_detector(cfg, is_fresh_train: bool = True):
    device = torch.device(cfg.RUNTIME.DEVICE)

    if is_fresh_train:
        # only load pretrained weights for fresh training, not for evaluation or resumed training
        dinotxt_weights = cfg.MODEL.DINO.DINOTXT_WEIGHTS
        backbone_weights = cfg.MODEL.DINO.BACKBONE_WEIGHTS
    else:
        dinotxt_weights = None
        backbone_weights = None
    
    backbone_model, head_model = dinov3_vitl16_dinotxt_tet1280d20h24l(
        pretrained=is_fresh_train,
        dinotxt_weights=dinotxt_weights,
        backbone_weights=backbone_weights,
        bpe_path_or_url=cfg.MODEL.DINO.BPE_PATH_OR_URL,
        return_seperate_vision_tower=True
    )   # keep only the vision model
    backbone = build_backbone(cfg, backbone_model=backbone_model)

    transformer = build_transformer(cfg)
    model = HOIDetector(cfg, backbone, head_model, transformer)

    weight_dict = {}
    losses = ["hoi_labels", "obj_labels", "sub_obj_boxes"]
    weight_dict["loss_hoi_labels"] = cfg.LOSS.COEFFS.HOI_LOSS_COEF
    weight_dict["loss_obj_ce"] = cfg.LOSS.COEFFS.OBJ_LOSS_COEF
    weight_dict["loss_sub_bbox"] = cfg.LOSS.COEFFS.BBOX_LOSS_COEF
    weight_dict["loss_obj_bbox"] = cfg.LOSS.COEFFS.BBOX_LOSS_COEF
    weight_dict["loss_sub_giou"] = cfg.LOSS.COEFFS.GIOU_LOSS_COEF
    weight_dict["loss_obj_giou"] = cfg.LOSS.COEFFS.GIOU_LOSS_COEF
    clip_soft_cfg = cfg.LOSS.get("CLIP_SOFT_LABEL", {})
    clip_soft_weight = float(_cfg_get(clip_soft_cfg, "WEIGHT", 0.0))
    if _cfg_bool(clip_soft_cfg, "ENABLED") and clip_soft_weight > 0:
        weight_dict["loss_hoi_clip_soft"] = clip_soft_weight
    sem_cfg = _cfg_get(cfg.MODEL, "SEMANTIC_ENHANCEMENT", {})
    sem_masked_weight = float(_cfg_get(sem_cfg, "MASKED_LOSS_WEIGHT", 0.0))
    if _cfg_bool(sem_cfg, "ENABLED") and sem_masked_weight > 0:
        weight_dict["loss_hoi_masked_semantic"] = sem_masked_weight
    sem_mask_recovery_weight = float(_cfg_get(sem_cfg, "MASK_RECOVERY_WEIGHT", 0.0))
    if _cfg_bool(sem_cfg, "ENABLED") and sem_mask_recovery_weight > 0:
        weight_dict["loss_mask_recovery"] = sem_mask_recovery_weight
    sem_vt_weight = float(
        _cfg_get(sem_cfg, "VT_CONSISTENCY_WEIGHT", _cfg_get(sem_cfg, "CONSISTENCY_WEIGHT", 0.0))
    )
    if _cfg_bool(sem_cfg, "ENABLED") and sem_vt_weight > 0:
        weight_dict["loss_visual_text_consistency"] = sem_vt_weight
    sem_proto_weight = float(_cfg_get(sem_cfg, "PROTO_CONSISTENCY_WEIGHT", 0.0))
    if _cfg_bool(sem_cfg, "ENABLED") and sem_proto_weight > 0:
        weight_dict["loss_global_proto_consistency"] = sem_proto_weight

    if cfg.MODEL.HOI.DEEP_SUPERVISION:
        num_dec_layers = model.num_dec_layers
        aux_weight_dict = {}
        for i in range(num_dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    matcher = build_matcher(cfg)
    criterion = SetCriterionHOI(cfg, matcher=matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {"hoi": PostProcessHOITriplet(cfg)}

    return model, criterion, postprocessors
