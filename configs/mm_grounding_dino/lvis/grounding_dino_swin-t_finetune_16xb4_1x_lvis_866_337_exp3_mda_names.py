_base_ = 'grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337.py'

# Exp-3: MDA-Augmented Class Names
# Augments text prompt with MDA attribute descriptions when confusable
# pairs co-occur. No auxiliary margin loss — the main classification loss
# itself becomes MDA-aware through richer text descriptions.

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
                   'all_label_index_map', 'label_remap_dict'))
]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(pipeline=train_pipeline)))
