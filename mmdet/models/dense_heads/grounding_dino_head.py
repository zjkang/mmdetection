# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import constant_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.losses import QualityFocalLoss
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList, reduce_mean
from ..layers import inverse_sigmoid
from ..mda import ConfusableSetManager, MDAEmbeddingCache, PlainMarginLoss
from .atss_vlfusion_head import convert_grounding_to_cls_scores
from .dino_head import DINOHead


class ContrastiveEmbed(nn.Module):
    """text visual ContrastiveEmbed layer.

    Args:
        max_text_len (int, optional): Maximum length of text.
        log_scale (Optional[Union[str, float]]):  The initial value of a
          learnable parameter to multiply with the similarity
          matrix to normalize the output.  Defaults to 0.0.
          - If set to 'auto', the similarity matrix will be normalized by
            a fixed value ``sqrt(d_c)`` where ``d_c`` is the channel number.
          - If set to 'none' or ``None``, there is no normalization applied.
          - If set to a float number, the similarity matrix will be multiplied
            by ``exp(log_scale)``, where ``log_scale`` is learnable.
        bias (bool, optional): Whether to add bias to the output.
          If set to ``True``, a learnable bias that is initialized as -4.6
          will be added to the output. Useful when training from scratch.
          Defaults to False.
    """

    def __init__(self,
                 max_text_len: int = 256,
                 log_scale: Optional[Union[str, float]] = None,
                 bias: bool = False):
        super().__init__()
        self.max_text_len = max_text_len
        self.log_scale = log_scale
        if isinstance(log_scale, float):
            self.log_scale = nn.Parameter(
                torch.Tensor([float(log_scale)]), requires_grad=True)
        elif log_scale not in ['auto', 'none', None]:
            raise ValueError(f'log_scale should be one of '
                             f'"auto", "none", None, but got {log_scale}')

        self.bias = None
        if bias:
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.bias = nn.Parameter(
                torch.Tensor([bias_value]), requires_grad=True)

    def forward(self, visual_feat: Tensor, text_feat: Tensor,
                text_token_mask: Tensor) -> Tensor:
        """Forward function.

        Args:
            visual_feat (Tensor): Visual features.
            text_feat (Tensor): Text features.
            text_token_mask (Tensor): A mask used for text feats.

        Returns:
            Tensor: Classification score.
        """
        res = visual_feat @ text_feat.transpose(-1, -2)
        if isinstance(self.log_scale, nn.Parameter):
            res = res * self.log_scale.exp()
        elif self.log_scale == 'auto':
            # NOTE: similar to the normalizer in self-attention
            res = res / math.sqrt(visual_feat.shape[-1])
        if self.bias is not None:
            res = res + self.bias
        res.masked_fill_(~text_token_mask[:, None, :], float('-inf'))

        new_res = torch.full((*res.shape[:-1], self.max_text_len),
                             float('-inf'),
                             device=res.device)
        new_res[..., :res.shape[-1]] = res

        return new_res


@MODELS.register_module()
class GroundingDINOHead(DINOHead):
    """Head of the Grounding DINO: Marrying DINO with Grounded Pre-Training for
    Open-Set Object Detection.

    Args:
        contrastive_cfg (dict, optional): Contrastive config that contains
          keys like ``max_text_len``. Defaults to dict(max_text_len=256).
        margin_config (dict, optional): Config for Pure Margin Loss.
          Keys:
            - confusable_index_path (str): Path to confusable_index.json.
            - margin (float): Margin value. Defaults to 0.2.
            - loss_weight (float): Weight for margin loss. Defaults to 0.5.
            - num_negatives (int): Number of confusable negatives. Defaults 3.
            - warmup_iters (int): Skip margin loss for first N iters. Default 500.
          If None, margin loss is disabled. Defaults to None.
    """

    def __init__(self,
                 contrastive_cfg=dict(max_text_len=256),
                 margin_config=None,
                 mda_margin_config=None,
                 mda_fused_config=None,
                 **kwargs):
        self.contrastive_cfg = contrastive_cfg
        self.max_text_len = contrastive_cfg.get('max_text_len', 256)
        self.margin_config = margin_config
        self.mda_margin_config = mda_margin_config
        self.mda_fused_config = mda_fused_config
        super().__init__(**kwargs)
        # Initialise margin loss components (after super().__init__)
        if margin_config is not None:
            self.confusable_mgr = ConfusableSetManager(
                margin_config['confusable_index_path'],
                num_negatives=margin_config.get('num_negatives', 3))
            self.margin_loss_fn = PlainMarginLoss(
                margin=margin_config.get('margin', 0.2))
            self.margin_loss_weight = margin_config.get('loss_weight', 0.5)
            self.margin_warmup_iters = margin_config.get('warmup_iters', 500)
            self._margin_iter = 0
        else:
            self.confusable_mgr = None

        # Initialise MDA margin loss components (Exp-2)
        if mda_margin_config is not None:
            self.mda_cache = MDAEmbeddingCache(
                mda_attributes_path=mda_margin_config[
                    'mda_attributes_path'],
                lvis_categories_path=mda_margin_config[
                    'lvis_categories_path'],
                confusable_index_path=mda_margin_config[
                    'confusable_index_path'],
                num_negatives=mda_margin_config.get('num_negatives', 3))
            self.mda_margin_loss_fn = PlainMarginLoss(
                margin=mda_margin_config.get('margin', 0.2))
            self.mda_margin_loss_weight = mda_margin_config.get(
                'loss_weight', 0.5)
            self.mda_margin_warmup_iters = mda_margin_config.get(
                'warmup_iters', 500)
            self._mda_margin_iter = 0
        else:
            self.mda_cache = None

        # B3: Fused MDA margin loss (MDA tokens in prompt, fused via encoder)
        if mda_fused_config is not None:
            self.fused_margin_loss_fn = PlainMarginLoss(
                margin=mda_fused_config.get('margin', 0.2))
            self.fused_margin_loss_weight = mda_fused_config.get(
                'loss_weight', 0.5)
            self.fused_margin_warmup_iters = mda_fused_config.get(
                'warmup_iters', 500)
            self._fused_margin_iter = 0
            self._fused_diag_interval = mda_fused_config.get(
                'diag_interval', 50)

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = ContrastiveEmbed(**self.contrastive_cfg)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        # NOTE: due to the fc_cls is a contrastive embedding and don't
        # have any trainable parameters,we do not need to copy it.
        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        gt_bboxes = gt_instances.bboxes

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # Major changes. The labels are 0-1 binary labels for each bbox
        # and text tokens.
        labels = gt_bboxes.new_full((num_bboxes, self.max_text_len),
                                    0,
                                    dtype=torch.float32)
        labels[pos_inds] = gt_instances.positive_maps[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
    ) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (List[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state,
                                                        memory_text,
                                                        text_token_mask)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (List[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).
            batch_data_samples (SampleList): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            InstanceList: Detection results of each image
                after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map
            for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references, memory_text, text_token_mask)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        batch_token_positive_maps: Optional[List[dict]] = None,
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor):  Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            batch_img_metas (List[Dict]): _description_
            batch_token_positive_maps (list[dict], Optional): Batch token
                positive map. Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            token_positive_maps = batch_token_positive_maps[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   token_positive_maps,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                token_positive_maps: dict,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            token_positive_maps (dict): Token positive map.
            img_meta (dict): Image meta info.
            rescale (bool, optional): If True, return boxes in original image
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
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']

        if token_positive_maps is not None:
            cls_score = convert_grounding_to_cls_scores(
                logits=cls_score.sigmoid()[None],
                positive_maps=[token_positive_maps])[0]
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            num_classes = cls_score.shape[-1]
            det_labels = indexes % num_classes
            bbox_index = indexes // num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            cls_score = cls_score.sigmoid()
            scores, _ = cls_score.max(-1)
            scores, indexes = scores.topk(max_per_img)
            bbox_pred = bbox_pred[indexes]
            det_labels = scores.new_zeros(scores.shape, dtype=torch.long)

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
        return results

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 4) and each `inter_reference` has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 4) with the
                last dimension arranged as (cx, cy, w, h).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references, memory_text, text_token_mask)
        self.text_masks = text_token_mask
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)

        # Pure Margin Loss (Exp-1)
        if self.confusable_mgr is not None:
            self._margin_iter += 1
            if self._margin_iter > self.margin_warmup_iters:
                # Split matching vs denoising queries; use last decoder layer
                (all_matching_cls, all_matching_bbox, _, _) = \
                    self.split_outputs(outs[0], outs[1], dn_meta)
                last_cls = all_matching_cls[-1]   # [bs, nq, max_text_len]
                last_bbox = all_matching_bbox[-1]  # [bs, nq, 4]
                loss_margin = self._compute_margin_loss(
                    last_cls, last_bbox, batch_gt_instances,
                    batch_img_metas, batch_data_samples)
                losses['loss_margin'] = loss_margin * self.margin_loss_weight

        # MDA Margin Loss (Exp-2)
        if self.mda_cache is not None and self.mda_cache.is_built:
            self._mda_margin_iter += 1
            if self._mda_margin_iter > self.mda_margin_warmup_iters:
                # Split matching vs denoising queries
                (all_matching_cls, all_matching_bbox, _, _) = \
                    self.split_outputs(outs[0], outs[1], dn_meta)
                last_cls = all_matching_cls[-1]
                last_bbox = all_matching_bbox[-1]
                # Get last layer hidden states (pre-ContrastiveEmbed)
                # hidden_states: [num_layers, bs, nq_total, 256]
                num_dn = dn_meta['num_denoising_queries'] \
                    if dn_meta is not None else 0
                last_hidden = hidden_states[-1][:, num_dn:, :]  # [bs,nq,256]
                loss_mda = self._compute_mda_margin_loss(
                    last_hidden, last_cls, last_bbox,
                    batch_gt_instances, batch_img_metas,
                    batch_data_samples)
                losses['loss_mda_margin'] = \
                    loss_mda * self.mda_margin_loss_weight

        # B3: Fused MDA Margin Loss (MDA embeddings from memory_text)
        if self.mda_fused_config is not None:
            self._fused_margin_iter += 1
            if self._fused_margin_iter > self.fused_margin_warmup_iters:
                (all_matching_cls, all_matching_bbox, _, _) = \
                    self.split_outputs(outs[0], outs[1], dn_meta)
                last_cls = all_matching_cls[-1]
                last_bbox = all_matching_bbox[-1]
                loss_fused = self._compute_fused_mda_margin_loss(
                    memory_text, text_token_mask,
                    last_cls, last_bbox,
                    batch_gt_instances, batch_img_metas,
                    batch_data_samples)
                losses['loss_fused_mda'] = \
                    loss_fused * self.fused_margin_loss_weight

        return losses

    def _compute_margin_loss(self, last_cls: Tensor, last_bbox: Tensor,
                             batch_gt_instances: InstanceList,
                             batch_img_metas: List[dict],
                             batch_data_samples: SampleList) -> Tensor:
        """Compute Pure Margin Loss over the last decoder layer's outputs.

        For each matched (query, GT) pair, penalises when confusable negative
        class scores are within ``margin`` of the positive class score.

        Uses ``all_label_index_map`` (original class id → token positions
        for ALL classes in the prompt, including negatives) and
        ``label_remap_dict`` (original id → prompt-local id) to map between
        the remapped GT labels and the original confusable_index ids.
        """
        all_pos_scores = []
        all_neg_scores = []

        for i in range(len(batch_gt_instances)):
            gt_instances = batch_gt_instances[i]
            img_meta = batch_img_metas[i]
            data_sample = batch_data_samples[i]
            cls_score_i = last_cls[i]   # [nq, max_text_len]
            bbox_pred_i = last_bbox[i]  # [nq, 4]

            # all_label_index_map: {orig_cls_id: (prompt_idx, [[s,e]])}
            all_lim = img_meta.get('all_label_index_map', None)
            remap = img_meta.get('label_remap_dict', None)
            if all_lim is None or remap is None:
                continue

            # Build inverse remap: prompt-local id → original class id
            inv_remap = {v: k for k, v in remap.items()}

            # Build full positive map for ALL classes in prompt using
            # the same tokenizer output stored in data_sample
            full_pm = data_sample.full_positive_map \
                if hasattr(data_sample, 'full_positive_map') else None

            # Re-run Hungarian matching (no grad — only to get indices)
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred_i.new_tensor(
                [img_w, img_h, img_w, img_h]).unsqueeze(0)
            bbox_xyxy = bbox_cxcywh_to_xyxy(bbox_pred_i) * factor

            with torch.no_grad():
                pred_inst = InstanceData(
                    scores=cls_score_i, bboxes=bbox_xyxy)
                assign_result = self.assigner.assign(
                    pred_instances=pred_inst,
                    gt_instances=gt_instances,
                    img_meta=img_meta)

            pos_inds = torch.nonzero(
                assign_result.gt_inds > 0,
                as_tuple=False).squeeze(-1).unique()
            if len(pos_inds) == 0:
                continue

            pos_gt_inds = assign_result.gt_inds[pos_inds] - 1
            gt_labels_matched = gt_instances.labels[pos_gt_inds]

            for q_idx, gt_lbl in zip(pos_inds, gt_labels_matched):
                # gt_lbl is prompt-local (remapped) id; convert to original
                local_id = gt_lbl.item()
                orig_cls_id = inv_remap.get(local_id, None)
                if orig_cls_id is None:
                    continue

                neg_ids = self.confusable_mgr.get_negatives(orig_cls_id)
                if not neg_ids:
                    continue

                # Score for positive class via full_positive_map
                if full_pm is not None and local_id in full_pm:
                    pos_mask = full_pm[local_id].to(cls_score_i.device)
                    active_pos = pos_mask > 0
                    if active_pos.sum() == 0:
                        continue
                    s_pos = cls_score_i[q_idx, active_pos].mean()
                else:
                    continue

                # Scores for confusable negatives using all_label_index_map
                neg_scores_list = []
                for neg_id in neg_ids:
                    if neg_id not in all_lim:
                        continue
                    neg_prompt_idx, _ = all_lim[neg_id]
                    # Look up in full_pm using prompt_idx
                    if full_pm is not None and neg_prompt_idx in full_pm:
                        neg_mask = full_pm[neg_prompt_idx].to(
                            cls_score_i.device)
                        active_neg = neg_mask > 0
                        if active_neg.sum() == 0:
                            continue
                        neg_scores_list.append(
                            cls_score_i[q_idx, active_neg].mean())

                if not neg_scores_list:
                    continue

                all_pos_scores.append(s_pos)
                all_neg_scores.append(
                    torch.stack(neg_scores_list))  # [K_avail]

        if not all_pos_scores:
            return last_cls.new_zeros(1).squeeze()

        total_loss = last_cls.new_zeros(1).squeeze()
        for s_pos, s_neg in zip(all_pos_scores, all_neg_scores):
            total_loss = total_loss + self.margin_loss_fn(
                s_pos.unsqueeze(0),
                s_neg.unsqueeze(0))
        return total_loss / len(all_pos_scores)

    def _compute_mda_margin_loss(
            self, last_hidden: Tensor, last_cls: Tensor, last_bbox: Tensor,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_data_samples: SampleList) -> Tensor:
        """Compute MDA Margin Loss using pre-computed MDA attribute embeddings.

        Instead of using class name token scores from the prompt, computes
        dot product between query embeddings and cached MDA attribute
        embeddings to get discriminative scores.

        Args:
            last_hidden: Decoder output [bs, nq, 256] (pre-ContrastiveEmbed).
            last_cls: Classification scores [bs, nq, max_text_len].
            last_bbox: Bbox predictions [bs, nq, 4].
            batch_gt_instances: GT instances per image.
            batch_img_metas: Image meta info per image.
            batch_data_samples: Full data samples.
        """
        # Get ContrastiveEmbed scaling parameters
        ce = self.cls_branches[-1]  # last layer's ContrastiveEmbed
        log_scale = ce.log_scale
        bias = ce.bias

        all_pos_scores = []
        all_neg_scores = []

        for i in range(len(batch_gt_instances)):
            gt_instances = batch_gt_instances[i]
            img_meta = batch_img_metas[i]
            cls_score_i = last_cls[i]   # [nq, max_text_len]
            bbox_pred_i = last_bbox[i]  # [nq, 4]
            hidden_i = last_hidden[i]   # [nq, 256]

            remap = img_meta.get('label_remap_dict', None)
            if remap is None:
                continue
            inv_remap = {v: k for k, v in remap.items()}

            # Hungarian matching
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred_i.new_tensor(
                [img_w, img_h, img_w, img_h]).unsqueeze(0)
            bbox_xyxy = bbox_cxcywh_to_xyxy(bbox_pred_i) * factor

            with torch.no_grad():
                pred_inst = InstanceData(
                    scores=cls_score_i, bboxes=bbox_xyxy)
                assign_result = self.assigner.assign(
                    pred_instances=pred_inst,
                    gt_instances=gt_instances,
                    img_meta=img_meta)

            pos_inds = torch.nonzero(
                assign_result.gt_inds > 0,
                as_tuple=False).squeeze(-1).unique()
            if len(pos_inds) == 0:
                continue

            pos_gt_inds = assign_result.gt_inds[pos_inds] - 1
            gt_labels_matched = gt_instances.labels[pos_gt_inds]

            for q_idx, gt_lbl in zip(pos_inds, gt_labels_matched):
                local_id = gt_lbl.item()
                orig_cls_id = inv_remap.get(local_id, None)
                if orig_cls_id is None:
                    continue

                neg_ids = self.mda_cache.get_negatives(orig_cls_id)
                if not neg_ids:
                    continue

                # Query embedding
                query_emb = hidden_i[q_idx]  # [256]

                neg_scores_list = []
                pos_score = None
                for neg_id in neg_ids:
                    pair = self.mda_cache.get_pair(orig_cls_id, neg_id)
                    if pair is None:
                        continue
                    attr_a_emb, attr_b_emb = pair  # each [256]
                    attr_a_emb = attr_a_emb.to(query_emb.device)
                    attr_b_emb = attr_b_emb.to(query_emb.device)

                    # Score = query · mda_emb (same as ContrastiveEmbed)
                    s_a = torch.dot(query_emb, attr_a_emb)
                    s_b = torch.dot(query_emb, attr_b_emb)

                    # Apply ContrastiveEmbed scaling then sigmoid
                    if isinstance(log_scale, nn.Parameter):
                        s_a = s_a * log_scale.exp()
                        s_b = s_b * log_scale.exp()
                    if bias is not None:
                        s_a = s_a + bias
                        s_b = s_b + bias
                    s_a = torch.sigmoid(s_a)
                    s_b = torch.sigmoid(s_b)

                    # s_a = score for positive class attribute
                    # s_b = score for negative class attribute
                    if pos_score is None:
                        pos_score = s_a
                    neg_scores_list.append(s_b)

                if pos_score is not None and neg_scores_list:
                    all_pos_scores.append(pos_score)
                    all_neg_scores.append(torch.stack(neg_scores_list))

        if not all_pos_scores:
            return last_cls.new_zeros(1).squeeze()

        total_loss = last_cls.new_zeros(1).squeeze()
        for s_pos, s_neg in zip(all_pos_scores, all_neg_scores):
            total_loss = total_loss + self.mda_margin_loss_fn(
                s_pos.unsqueeze(0),
                s_neg.unsqueeze(0))
        return total_loss / len(all_pos_scores)

    def _compute_fused_mda_margin_loss(
            self, memory_text: Tensor, text_token_mask: Tensor,
            last_cls: Tensor, last_bbox: Tensor,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_data_samples: SampleList) -> Tensor:
        """B3: Fused MDA Margin Loss using encoder-fused MDA embeddings.

        MDA attribute texts are appended to the prompt as separate
        sub-sentences, go through BERT + encoder cross-attention, and
        their fused embeddings are extracted from memory_text.
        Score = query · fused_mda_emb (same space as cls_score).

        Args:
            memory_text: Encoder-fused text features [bs, num_tokens, 256].
            text_token_mask: Valid token mask [bs, num_tokens].
            last_cls: Classification scores [bs, nq, max_text_len].
            last_bbox: Bbox predictions [bs, nq, 4].
            batch_gt_instances: GT instances per image.
            batch_img_metas: Image meta info per image.
            batch_data_samples: Full data samples.
        """
        all_pos_scores = []
        all_neg_scores = []
        # Diagnostics
        all_pos_raw = []
        all_neg_raw = []
        all_cosines = []

        for i in range(len(batch_gt_instances)):
            gt_instances = batch_gt_instances[i]
            img_meta = batch_img_metas[i]
            data_sample = batch_data_samples[i]
            cls_score_i = last_cls[i]
            bbox_pred_i = last_bbox[i]
            mem_text_i = memory_text[i]  # [num_tokens, 256]

            remap = img_meta.get('label_remap_dict', None)
            if remap is None:
                continue
            inv_remap = {v: k for k, v in remap.items()}

            # Get MDA token indices (char→token already converted)
            mda_indices = getattr(data_sample, 'mda_token_indices', None)
            if not mda_indices:
                continue

            # Get all_label_index_map for class name token positions
            all_lim = img_meta.get('all_label_index_map', None)

            # Hungarian matching
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred_i.new_tensor(
                [img_w, img_h, img_w, img_h]).unsqueeze(0)
            bbox_xyxy = bbox_cxcywh_to_xyxy(bbox_pred_i) * factor

            with torch.no_grad():
                pred_inst = InstanceData(
                    scores=cls_score_i, bboxes=bbox_xyxy)
                assign_result = self.assigner.assign(
                    pred_instances=pred_inst,
                    gt_instances=gt_instances,
                    img_meta=img_meta)

            pos_inds = torch.nonzero(
                assign_result.gt_inds > 0,
                as_tuple=False).squeeze(-1).unique()
            if len(pos_inds) == 0:
                continue

            pos_gt_inds = assign_result.gt_inds[pos_inds] - 1
            gt_labels_matched = gt_instances.labels[pos_gt_inds]

            for q_idx, gt_lbl in zip(pos_inds, gt_labels_matched):
                local_id = gt_lbl.item()
                orig_cls_id = inv_remap.get(local_id, None)
                if orig_cls_id is None:
                    continue

                # Find MDA pairs involving this class
                for pair_key, tok_idx in mda_indices.items():
                    cls_idx, neg_idx = pair_key
                    if cls_idx != orig_cls_id:
                        continue

                    pos_tok_s, pos_tok_e = tok_idx['pos']
                    neg_tok_s, neg_tok_e = tok_idx['neg']

                    # Extract fused MDA embeddings (mean pool over tokens)
                    pos_emb = mem_text_i[pos_tok_s:pos_tok_e + 1].mean(dim=0)
                    neg_emb = mem_text_i[neg_tok_s:neg_tok_e + 1].mean(dim=0)

                    # Query embedding from cls_score perspective:
                    # cls_score = query · memory_text already computed.
                    # We need to get the score for the MDA token positions.
                    # Use the pre-computed cls_score at MDA token positions.
                    s_pos = cls_score_i[q_idx, pos_tok_s:pos_tok_e + 1].mean()
                    s_neg = cls_score_i[q_idx, neg_tok_s:neg_tok_e + 1].mean()

                    # Diagnostics: raw scores and cosine similarity
                    all_pos_raw.append(s_pos.item())
                    all_neg_raw.append(s_neg.item())

                    # Cosine sim between MDA emb and class name emb
                    if all_lim is not None and orig_cls_id in all_lim:
                        full_pm = data_sample.full_positive_map
                        pm_idx = all_lim[orig_cls_id][0]
                        if pm_idx in full_pm:
                            cls_tok_positions = torch.nonzero(
                                full_pm[pm_idx],
                                as_tuple=True)[0]
                            if len(cls_tok_positions) > 0:
                                with torch.no_grad():
                                    cls_emb = mem_text_i[
                                        cls_tok_positions].mean(dim=0)
                                    cos = torch.nn.functional.cosine_similarity(
                                        pos_emb.detach().unsqueeze(0),
                                        cls_emb.unsqueeze(0)).item()
                                all_cosines.append(cos)

                    all_pos_scores.append(s_pos)
                    all_neg_scores.append(s_neg)

        # Diagnostic logging
        if all_pos_raw and self._fused_margin_iter % \
                self._fused_diag_interval == 0:
            import logging
            logger = logging.getLogger('mmdet')
            n = len(all_pos_raw)
            ratio = sum(1 for p, ng in zip(all_pos_raw, all_neg_raw)
                        if p > ng) / n
            avg_pos = sum(all_pos_raw) / n
            avg_neg = sum(all_neg_raw) / n
            msg = (f'[iter {self._fused_margin_iter}] '
                   f'MDA signal: pos={avg_pos:.3f}, neg={avg_neg:.3f}, '
                   f'ratio={ratio:.2%}, n_pairs={n}')
            if all_cosines:
                avg_cos = sum(all_cosines) / len(all_cosines)
                min_cos = min(all_cosines)
                max_cos = max(all_cosines)
                msg += (f' | MDA-classname cosine: '
                        f'mean={avg_cos:.3f}, '
                        f'min={min_cos:.3f}, max={max_cos:.3f}')
            logger.info(msg)

        if not all_pos_scores:
            return last_cls.new_zeros(1).squeeze()

        total_loss = last_cls.new_zeros(1).squeeze()
        for s_pos, s_neg in zip(all_pos_scores, all_neg_scores):
            total_loss = total_loss + self.fused_margin_loss_fn(
                s_pos.unsqueeze(0),
                s_neg.unsqueeze(0))
        return total_loss / len(all_pos_scores)

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        with torch.no_grad():
            cls_reg_targets = self.get_targets(cls_scores_list,
                                               bbox_preds_list,
                                               batch_gt_instances,
                                               batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert (self.text_masks.dim() == 2)
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
        cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()

        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError(
                'QualityFocalLoss for GroundingDINOHead is not supported yet.')
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
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
        return loss_cls, loss_bbox, loss_iou

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert (self.text_masks.dim() == 2)
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, dn_cls_scores.size(1), 1)
        cls_scores = torch.masked_select(dn_cls_scores, text_mask).contiguous()
        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)
        # =======================

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                raise NotImplementedError('QualityFocalLoss is not supported')
            else:
                loss_cls = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2
        # label targets
        # this change
        labels = gt_bboxes.new_full((num_denoising_queries, self.max_text_len),
                                    0,
                                    dtype=torch.float32)
        labels[pos_inds] = gt_instances.positive_maps[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)
