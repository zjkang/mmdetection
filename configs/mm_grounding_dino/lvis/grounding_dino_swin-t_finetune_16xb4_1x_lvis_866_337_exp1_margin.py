_base_ = 'grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337.py'

# Exp-1: Pure Margin Loss
# Inherits everything from the 866/337 OV-LVIS baseline config and only
# adds margin_config to the bbox_head.
model = dict(
    bbox_head=dict(
        margin_config=dict(
            confusable_index_path='data/mda/confusable_index.json',
            margin=0.2,
            loss_weight=0.5,
            num_negatives=3,
            warmup_iters=500,
        )))
