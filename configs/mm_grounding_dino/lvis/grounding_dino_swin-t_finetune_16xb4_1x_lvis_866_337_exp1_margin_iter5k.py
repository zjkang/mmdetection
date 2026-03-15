# Exp-1 Pure Margin: 5000 iters on single GPU (batch_size=4)
_base_ = 'grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337_exp1_margin.py'

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
        dataset=dict(pipeline=_base_.train_pipeline)),
)

optim_wrapper = dict(
    optimizer=dict(lr=0.000003125),
)

param_scheduler = []

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
