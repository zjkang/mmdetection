_base_ = 'grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337.py'

# B3: Fused MDA Margin Loss
# MDA attribute texts are appended to the prompt as separate sub-sentences,
# go through the full BERT + encoder fusion pipeline, and their fused
# embeddings from memory_text are used for margin loss.
# No distribution shift: class name tokens are unchanged; MDA sub-sentences
# are isolated in BERT attention (sub-sentence masking).
model = dict(
    bbox_head=dict(
        mda_fused_config=dict(
            margin=0.2,
            loss_weight=0.5,
            warmup_iters=500,
            diag_interval=50,
        )))

# Override train_pipeline to inject confusable negatives + MDA sub-sentences
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        label_map_file='data/coco/annotations/lvis_v1_label_map_norare.json',
        max_tokens=256,
        confusable_index_path='data/mda/confusable_index.json',
        mda_attributes_path='data/mda/mda_attributes.json',
        lvis_categories_path='data/mda/lvis_categories.json',
        mda_max_attr_tokens=6,
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode',
                   'all_label_index_map', 'label_remap_dict',
                   'mda_token_spans'))
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(pipeline=train_pipeline)))
