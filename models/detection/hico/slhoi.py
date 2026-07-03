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


class HOISemanticWordMaskEnhancer(nn.Module):
    """HOI prompt word masking branch for masked semantic recovery and SCC."""

    def __init__(self, feature_dim, dim_feedforward=4096, dropout=0.1, temperature=1.0):
        super().__init__()
        self.temperature = max(float(temperature), 1e-6)
        self.raw_prompt_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )
        self.masked_recover = nn.Sequential(
            nn.Linear(feature_dim * 2, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, feature_dim),
        )
        self.raw_norm = nn.LayerNorm(feature_dim)
        self.ref_norm = nn.LayerNorm(feature_dim)

    def _prompt_context(self, pair_feature, text_embedding):
        logits = torch.matmul(pair_feature, text_embedding.t()) / self.temperature
        return torch.softmax(logits, dim=-1).matmul(text_embedding)

    def forward(self, pair_feature, raw_text_embedding, masked_text_embedding, logit_scale):
        raw_text_embedding = F.normalize(
            raw_text_embedding.to(device=pair_feature.device, dtype=pair_feature.dtype), dim=-1
        )
        masked_text_embedding = F.normalize(
            masked_text_embedding.to(device=pair_feature.device, dtype=pair_feature.dtype), dim=-1
        )
        if raw_text_embedding.shape != masked_text_embedding.shape:
            raise ValueError(
                f"HOI masked prompt bank dim mismatch: raw={raw_text_embedding.shape}, "
                f"masked={masked_text_embedding.shape}"
            )
        if pair_feature.shape[-1] != raw_text_embedding.shape[-1]:
            raise ValueError(
                f"HOI semantic feature dim mismatch: pair_feature={pair_feature.shape}, "
                f"text_embedding={raw_text_embedding.shape}"
            )

        pair_feature = F.normalize(pair_feature, dim=-1)
        raw_context = self._prompt_context(pair_feature, raw_text_embedding)
        masked_context = self._prompt_context(pair_feature, masked_text_embedding)

        h_raw = self.raw_norm(pair_feature + self.raw_prompt_proj(raw_context))
        h_ref = self.ref_norm(pair_feature + self.masked_recover(torch.cat([pair_feature, masked_context], dim=-1)))
        h_raw = F.normalize(h_raw, dim=-1)
        h_ref = F.normalize(h_ref, dim=-1)

        loss_sem = (1.0 - F.cosine_similarity(h_raw, h_ref, dim=-1)).mean()
        ref_logits = logit_scale * torch.matmul(h_ref, raw_text_embedding.t())
        return ref_logits, loss_sem


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
        except ImportError as exc:
            raise ImportError(
                "ViTPose integration requires MMPose. Install mmpose/mmcv/mmengine "
                "and set MODEL.VITPOSE.CONFIG / MODEL.VITPOSE.CHECKPOINT."
            ) from exc
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
            self.semantic_enhancer = HOISemanticWordMaskEnhancer(
                feature_dim=2 * cfg.MODEL.DINO.EMBED_DIM,
                dim_feedforward=int(_cfg_get(sem_cfg, "DIM_FEEDFORWARD", 4096)),
                dropout=float(_cfg_get(sem_cfg, "DROPOUT", 0.1)),
                temperature=float(_cfg_get(sem_cfg, "TEMPERATURE", 1.0)),
            )
        else:
            self.semantic_enhancer = None

        ps_cfg = _cfg_get(cfg.MODEL, "POSE_SPATIAL_INTERACTION", {})
        self.use_pose_spatial_interaction = _cfg_bool(ps_cfg, "ENABLED", False)
        self.pose_spatial_detach_boxes = _cfg_bool(ps_cfg, "DETACH_BOXES", True)
        self.pose_spatial_num_keypoints = int(_cfg_get(ps_cfg, "NUM_KEYPOINTS", 6))
        if self.use_pose_spatial_interaction:
            self.pose_spatial_encoder = PoseSpatialInteractionEncoder(
                hidden_dim=hidden_dim,
                dino_embed_dim=dino_embed_dim,
                num_keypoints=self.pose_spatial_num_keypoints,
                dropout=float(_cfg_get(ps_cfg, "DROPOUT", 0.1)),
                init_scale=float(_cfg_get(ps_cfg, "INIT_SCALE", 0.1)),
            )
        else:
            self.pose_spatial_encoder = None

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

        pose_spatial_hidden = None
        pose_spatial_dino = None
        if self.pose_spatial_encoder is not None:
            ps_sub_boxes = outputs_sub_coord.detach() if self.pose_spatial_detach_boxes else outputs_sub_coord
            ps_obj_boxes = outputs_obj_coord.detach() if self.pose_spatial_detach_boxes else outputs_obj_coord
            if pose_keypoints is None and self.vitpose_estimator is not None:
                pose_keypoints = self.vitpose_estimator(samples, ps_sub_boxes.detach())
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
            use_eval_hoi_bank = (
                self.cfg.INPUT.DATASET_FILE == "hico"
                and self.cfg.ZERO_SHOT.TYPE != "default"
                and (self.cfg.RUNTIME.EVAL or not self.training)
            )
            if use_eval_hoi_bank:
                outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
            else:
                outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)
        else:
            raise ValueError("Please use DINO.txt label for HOI classification!")

        semantic_consistency_loss = None
        outputs_hoi_masked_class = None
        if self.semantic_enhancer is not None and self.training:
            outputs_hoi_masked_class, semantic_consistency_loss = self.semantic_enhancer(
                inter_hs,
                self.visual_projection.weight,
                self.hoi_masked_embedding_train,
                logit_scale,
            )

        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_obj_logits': outputs_obj_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if semantic_consistency_loss is not None:
            out['loss_semantic_consistency'] = semantic_consistency_loss
        if outputs_hoi_masked_class is not None:
            out['pred_hoi_masked_logits'] = outputs_hoi_masked_class[-1]

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
    sem_weight = float(_cfg_get(sem_cfg, "CONSISTENCY_WEIGHT", 0.0))
    if _cfg_bool(sem_cfg, "ENABLED") and sem_weight > 0:
        weight_dict["loss_semantic_consistency"] = sem_weight
    sem_masked_weight = float(_cfg_get(sem_cfg, "MASKED_LOSS_WEIGHT", 0.0))
    if _cfg_bool(sem_cfg, "ENABLED") and sem_masked_weight > 0:
        weight_dict["loss_hoi_masked_semantic"] = sem_masked_weight

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
