import json
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


class MDAEmbeddingCache:
    """Pre-computes and caches MDA attribute text embeddings.

    For each confusable pair (class_a, class_b), encodes the distinguishing
    attribute texts through BERT + text_feat_map to produce 256-dim vectors.

    Args:
        mda_attributes_path (str): Path to mda_attributes.json.
        lvis_categories_path (str): Path to lvis_categories.json.
        confusable_index_path (str): Path to confusable_index.json.
        num_negatives (int): Number of confusable negatives per class.
    """

    def __init__(self,
                 mda_attributes_path: str,
                 lvis_categories_path: str,
                 confusable_index_path: str,
                 num_negatives: int = 3):
        # Load MDA attributes: "class_a|class_b" -> {attr_a, attr_b}
        with open(mda_attributes_path, 'r') as f:
            self._mda_attrs = json.load(f)

        # Build cont_idx -> name and name -> cont_idx mappings
        with open(lvis_categories_path, 'r') as f:
            cats_data = json.load(f)
        categories = cats_data['categories']
        self._idx_to_name = {}
        self._name_to_idx = {}
        for cat in categories:
            idx = cat['cont_idx']
            name = cat['name']
            self._idx_to_name[idx] = name
            self._name_to_idx[name] = idx

        # Load confusable index for iteration
        with open(confusable_index_path, 'r') as f:
            raw = json.load(f)
        self._confusable_map = {
            int(k): v[:num_negatives] for k, v in raw.items()
        }

        self._cache: Dict[Tuple[int, int], Tuple[Tensor, Tensor]] = {}
        self.is_built = False

    def build_cache(self, language_model, text_feat_map, device) -> None:
        """Encode all MDA attribute texts and cache embeddings.

        Args:
            language_model: The BERT language model from GroundingDINO.
            text_feat_map: Linear projection (768 -> 256).
            device: Target device.
        """
        if self.is_built:
            return

        # Collect all unique attribute texts
        texts_to_encode = []
        text_keys = []  # (pair_key, 'attr_a' or 'attr_b')

        for cls_idx, neg_indices in self._confusable_map.items():
            cls_name = self._idx_to_name.get(cls_idx)
            if cls_name is None:
                continue
            for neg_idx in neg_indices:
                neg_name = self._idx_to_name.get(neg_idx)
                if neg_name is None:
                    continue
                pair_key = f"{cls_name}|{neg_name}"
                if pair_key not in self._mda_attrs:
                    continue
                attrs = self._mda_attrs[pair_key]
                texts_to_encode.append(attrs['attr_a'])
                text_keys.append(((cls_idx, neg_idx), 'attr_a'))
                texts_to_encode.append(attrs['attr_b'])
                text_keys.append(((cls_idx, neg_idx), 'attr_b'))

        if not texts_to_encode:
            self.is_built = True
            return

        # Encode in batches to avoid OOM
        batch_size = 64
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts_to_encode), batch_size):
                batch_texts = texts_to_encode[i:i + batch_size]
                text_dict = language_model(batch_texts)
                embedded = text_dict['embedded']  # [B, seq_len, 768]
                masks = text_dict['masks']  # [B, seq_len]

                # Project 768 -> 256
                projected = text_feat_map(embedded)  # [B, seq_len, 256]

                # Mean pool over valid tokens
                if masks.dim() == 2:
                    mask_float = masks.float().unsqueeze(-1)  # [B, seq, 1]
                    pooled = (projected * mask_float).sum(dim=1) / \
                        mask_float.sum(dim=1).clamp(min=1)  # [B, 256]
                else:
                    pooled = projected.mean(dim=1)

                all_embeddings.append(pooled)

        all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, 256]

        # Store in cache
        pair_embeddings: Dict[Tuple[int, int], dict] = {}
        for idx, (key_info, attr_type) in enumerate(text_keys):
            pair = key_info
            if pair not in pair_embeddings:
                pair_embeddings[pair] = {}
            pair_embeddings[pair][attr_type] = all_embeddings[idx]

        for pair, embs in pair_embeddings.items():
            if 'attr_a' in embs and 'attr_b' in embs:
                self._cache[pair] = (embs['attr_a'], embs['attr_b'])

        self.is_built = True
        print(f"MDAEmbeddingCache: cached {len(self._cache)} pairs")

    def get_pair(self, cls_idx: int,
                 neg_idx: int) -> Optional[Tuple[Tensor, Tensor]]:
        """Get cached MDA embeddings for a confusable pair.

        Returns:
            Tuple of (attr_a_embed, attr_b_embed) each [256], or None.
        """
        return self._cache.get((cls_idx, neg_idx))

    def get_negatives(self, cls_idx: int):
        """Get confusable negative indices for a class."""
        return self._confusable_map.get(cls_idx, [])
