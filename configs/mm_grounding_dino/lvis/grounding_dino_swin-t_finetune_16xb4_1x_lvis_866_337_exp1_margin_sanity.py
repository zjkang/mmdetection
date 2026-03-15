# Sanity check: Exp-1 with iter-based loop (20 iters only)
_base_ = 'grounding_dino_swin-t_finetune_16xb4_1x_lvis_866_337_exp1_margin.py'

# Replace train_cfg entirely so IterBasedTrainLoop gets no max_epochs
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=20,
    val_interval=999,
    val_begin=999,
)

# Do not load backbone from URL in init_weights; use runner load_from only (local checkpoint)
model = dict(backbone=dict(init_cfg=None))

# Print every iter so we can see losses in the 20-iter sanity run
default_hooks = dict(logger=dict(type='LoggerHook', interval=1))
log_processor = dict(window_size=1)
