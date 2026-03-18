# Exp-3a MDA Names (p=1.0, always augment): 5000 iters, single GPU
_base_ = 'grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337.py'

# Override train_pipeline with MDA augmentation (mda_aug_prob=1.0)
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
        mda_aug_prob=1.0,
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode',
                   'all_label_index_map', 'label_remap_dict'))
]

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=5000,
    val_interval=5000,
    val_begin=5000,
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        dataset=dict(pipeline=train_pipeline)),
)

optim_wrapper = dict(
    optimizer=dict(lr=0.000003125),
)

param_scheduler = []

model = dict(test_cfg=dict(chunked_size=40))

val_dataloader = dict(batch_size=1)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2500,
        max_keep_ckpts=2,
        save_best='lvis_fixed_ap/AP',
        rule='greater'),
)
