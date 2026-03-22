# Copyright (c) OpenMMLab. All rights reserved.
import json

from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes

try:
    from transformers import AutoTokenizer
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

import random
import re

import numpy as np


def clean_name(name):
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    name = name.lower()
    return name


def check_for_positive_overflow(gt_bboxes, gt_labels, text, tokenizer,
                                max_tokens):
    # Check if we have too many positive labels
    # generate a caption by appending the positive labels
    positive_label_list = np.unique(gt_labels).tolist()
    # random shuffule so we can sample different annotations
    # at different epochs
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for index, label in enumerate(positive_label_list):

        label_text = clean_name(text[str(label)]) + '. '

        tokenized = tokenizer.tokenize(label_text)

        length += len(tokenized)

        if length > max_tokens:
            break
        else:
            kept_lables.append(label)

    keep_box_index = []
    keep_gt_labels = []
    for i in range(len(gt_labels)):
        if gt_labels[i] in kept_lables:
            keep_box_index.append(i)
            keep_gt_labels.append(gt_labels[i])

    return gt_bboxes[keep_box_index], np.array(
        keep_gt_labels, dtype=np.int64), length


def generate_senetence_given_labels(positive_label_list, negative_label_list,
                                    text):
    label_to_positions = {}

    label_list = negative_label_list + positive_label_list

    random.shuffle(label_list)

    pheso_caption = ''

    label_remap_dict = {}
    # Map from original class id to prompt-local index for ALL classes
    all_label_index_map = {}
    for index, label in enumerate(label_list):

        start_index = len(pheso_caption)

        pheso_caption += clean_name(text[str(label)])

        end_index = len(pheso_caption)

        # Record token positions for ALL classes (pos + neg)
        all_label_index_map[int(label)] = (index, [[start_index, end_index]])

        if label in positive_label_list:
            label_to_positions[index] = [[start_index, end_index]]
            label_remap_dict[int(label)] = index

        # if index != len(label_list) - 1:
        #     pheso_caption += '. '
        pheso_caption += '. '

    return label_to_positions, pheso_caption, label_remap_dict, \
        all_label_index_map


@TRANSFORMS.register_module()
class RandomSamplingNegPos(BaseTransform):

    def __init__(self,
                 tokenizer_name,
                 num_sample_negative=85,
                 max_tokens=256,
                 full_sampling_prob=0.5,
                 label_map_file=None,
                 confusable_index_path=None,
                 mda_attributes_path=None,
                 lvis_categories_path=None,
                 mda_max_attr_tokens=6,
                 mda_aug_prob=0.5):
        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_sample_negative = num_sample_negative
        self.full_sampling_prob = full_sampling_prob
        self.max_tokens = max_tokens
        self.label_map = None
        if label_map_file:
            with open(label_map_file, 'r') as file:
                self.label_map = json.load(file)
        self.confusable_index = None
        if confusable_index_path:
            with open(confusable_index_path, 'r') as f:
                raw = json.load(f)
            # keys are strings, convert to int→list[int]
            self.confusable_index = {
                int(k): v for k, v in raw.items()
            }

        # MDA attribute augmentation (Exp-3)
        # Maps (cont_idx_a, cont_idx_b) → (attr_for_a, attr_for_b)
        self.mda_pair_attrs = None
        self.mda_aug_prob = mda_aug_prob
        if mda_attributes_path and lvis_categories_path:
            self._build_mda_lookup(
                mda_attributes_path, lvis_categories_path,
                mda_max_attr_tokens)

    def _build_mda_lookup(self, mda_attributes_path, lvis_categories_path,
                          max_attr_tokens):
        """Build pair-indexed MDA attribute lookup with pre-truncated text."""
        with open(mda_attributes_path, 'r') as f:
            mda_attrs = json.load(f)
        with open(lvis_categories_path, 'r') as f:
            cats_data = json.load(f)

        name_to_idx = {}
        for cat in cats_data['categories']:
            name_to_idx[cat['name']] = cat['cont_idx']

        self.mda_pair_attrs = {}
        for pair_key, attrs in mda_attrs.items():
            parts = pair_key.split('|')
            if len(parts) != 2:
                continue
            name_a, name_b = parts
            idx_a = name_to_idx.get(name_a)
            idx_b = name_to_idx.get(name_b)
            if idx_a is None or idx_b is None:
                continue

            # Truncate attribute text to max_attr_tokens BERT tokens
            attr_a = self._truncate_attr(attrs['attr_a'], max_attr_tokens)
            attr_b = self._truncate_attr(attrs['attr_b'], max_attr_tokens)

            self.mda_pair_attrs[(idx_a, idx_b)] = (attr_a, attr_b)
            self.mda_pair_attrs[(idx_b, idx_a)] = (attr_b, attr_a)

        print(f'MDA augmentation: loaded {len(self.mda_pair_attrs)} '
              f'pair entries (max_attr_tokens={max_attr_tokens})')

    def _truncate_attr(self, attr_text, max_tokens):
        """Truncate attribute text to fit within max BERT tokens."""
        attr_text = attr_text.lower().strip()
        tokens = self.tokenizer.tokenize(attr_text)
        if len(tokens) <= max_tokens:
            return attr_text
        # Decode truncated tokens back to text
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens[:max_tokens])
        return self.tokenizer.decode(token_ids).strip()

    def _collect_mda_pairs(self, all_label_ids):
        """Collect MDA attribute pairs for confusable classes in the prompt.

        For each positive class with a confusable partner also in the prompt,
        collects the (cls_idx, neg_idx, attr_for_cls, attr_for_neg) tuple.
        Used by B3 to append MDA sub-sentences to the prompt.

        Returns:
            list of (cls_idx, neg_idx, attr_cls, attr_neg) tuples.
        """
        if self.mda_pair_attrs is None:
            return []

        all_ids_set = set(int(x) for x in all_label_ids)
        seen_pairs = set()
        mda_pairs = []

        for lid in all_ids_set:
            for partner in all_ids_set:
                if lid == partner:
                    continue
                pair = (lid, partner)
                if pair in self.mda_pair_attrs and pair not in seen_pairs:
                    attr_cls, attr_neg = self.mda_pair_attrs[pair]
                    seen_pairs.add(pair)
                    seen_pairs.add((partner, lid))  # deduplicate reverse
                    mda_pairs.append((lid, partner, attr_cls, attr_neg))

        return mda_pairs

    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            return self.vg_aug(results)
        else:
            return self.od_aug(results)

    def vg_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        text = results['text'].lower().strip()
        if not text.endswith('.'):
            text = text + '. '

        phrases = results['phrases']
        # TODO: add neg
        positive_label_list = np.unique(gt_labels).tolist()
        label_to_positions = {}
        for label in positive_label_list:
            label_to_positions[label] = phrases[label]['tokens_positive']

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = text
        results['tokens_positive'] = label_to_positions
        return results

    def od_aug(self, results):
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']

        if 'text' not in results:
            assert self.label_map is not None
            text = self.label_map
        else:
            text = results['text']

        original_box_num = len(gt_labels)
        # If the category name is in the format of 'a/b' (in object365),
        # we randomly select one of them.
        for key, value in text.items():
            if '/' in value:
                text[key] = random.choice(value.split('/')).strip()

        gt_bboxes, gt_labels, positive_caption_length = \
            check_for_positive_overflow(gt_bboxes, gt_labels,
                                        text, self.tokenizer, self.max_tokens)

        if len(gt_bboxes) < original_box_num:
            print('WARNING: removed {} boxes due to positive caption overflow'.
                  format(original_box_num - len(gt_bboxes)))

        valid_negative_indexes = list(text.keys())

        positive_label_list = np.unique(gt_labels).tolist()
        full_negative = self.num_sample_negative

        if full_negative > len(valid_negative_indexes):
            full_negative = len(valid_negative_indexes)

        outer_prob = random.random()

        if outer_prob < self.full_sampling_prob:
            # c. probability_full: add both all positive and all negatives
            num_negatives = full_negative
        else:
            if random.random() < 1.0:
                num_negatives = np.random.choice(max(1, full_negative)) + 1
            else:
                num_negatives = full_negative

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)

            for i in np.random.choice(
                    valid_negative_indexes, size=num_negatives, replace=False):
                if int(i) not in positive_label_list:
                    negative_label_list.add(i)

        # Force confusable negatives into the prompt so margin loss can fire
        if self.confusable_index is not None:
            valid_neg_set = set(int(k) for k in valid_negative_indexes)
            for gt_lbl in positive_label_list:
                for neg_id in self.confusable_index.get(gt_lbl, []):
                    if neg_id not in positive_label_list \
                            and neg_id in valid_neg_set:
                        negative_label_list.add(str(neg_id))

        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)
        random.shuffle(negative_label_list)

        negative_max_length = self.max_tokens - positive_caption_length
        screened_negative_label_list = []

        for negative_label in negative_label_list:
            label_text = clean_name(text[str(negative_label)]) + '. '

            tokenized = self.tokenizer.tokenize(label_text)

            negative_max_length -= len(tokenized)

            if negative_max_length > 0:
                screened_negative_label_list.append(negative_label)
            else:
                break
        negative_label_list = screened_negative_label_list

        # Generate class name prompt (unchanged — no MDA in class names)
        label_to_positions, pheso_caption, label_remap_dict, \
            all_label_index_map = \
            generate_senetence_given_labels(positive_label_list,
                                            negative_label_list, text)

        # B3: Append MDA attribute sub-sentences AFTER class names
        # These are separate sub-sentences that don't modify class names.
        # Format: "... classN . mda_attr_1 . mda_attr_2 . ..."
        # Each MDA attr gets its own sub-sentence (isolated in BERT attn).
        mda_token_spans = {}  # "cls_idx,neg_idx" → {'pos': [s,e], 'neg': [s,e]}
        if self.mda_pair_attrs is not None:
            all_labels = [int(x) for x in negative_label_list] + \
                positive_label_list
            mda_pairs = self._collect_mda_pairs(all_labels)

            # Compute remaining token budget
            current_tokens = len(self.tokenizer.tokenize(pheso_caption))
            remaining = self.max_tokens - current_tokens - 5  # safety margin

            for cls_idx, neg_idx, attr_cls, attr_neg in mda_pairs:
                # Each MDA sub-sentence: "attr_text . "
                attr_cls_text = attr_cls.lower().strip()
                attr_neg_text = attr_neg.lower().strip()
                candidate = f'{attr_cls_text} . {attr_neg_text} . '
                candidate_tokens = len(self.tokenizer.tokenize(candidate))

                if candidate_tokens > remaining:
                    break  # no more budget

                # Record char positions for positive attr
                pos_start = len(pheso_caption)
                pheso_caption += attr_cls_text
                pos_end = len(pheso_caption)
                pheso_caption += ' . '

                # Record char positions for negative attr
                neg_start = len(pheso_caption)
                pheso_caption += attr_neg_text
                neg_end = len(pheso_caption)
                pheso_caption += ' . '

                mda_token_spans[f'{cls_idx},{neg_idx}'] = {
                    'pos': [pos_start, pos_end],
                    'neg': [neg_start, neg_end],
                }
                remaining -= candidate_tokens

        # label remap
        if len(gt_labels) > 0:
            gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels

        results['text'] = pheso_caption
        results['tokens_positive'] = label_to_positions
        # Pass original-id → token positions for ALL classes in prompt
        # (used by margin loss to look up confusable negatives)
        results['all_label_index_map'] = all_label_index_map
        # Also pass original-id → prompt-local remap for GT classes
        results['label_remap_dict'] = label_remap_dict
        # B3: MDA attribute char spans for fused margin loss
        results['mda_token_spans'] = mda_token_spans

        return results


@TRANSFORMS.register_module()
class LoadTextAnnotations(BaseTransform):

    def transform(self, results: dict) -> dict:
        if 'phrases' in results:
            tokens_positive = [
                phrase['tokens_positive']
                for phrase in results['phrases'].values()
            ]
            results['tokens_positive'] = tokens_positive
        else:
            text = results['text']
            results['text'] = list(text.values())
        return results
