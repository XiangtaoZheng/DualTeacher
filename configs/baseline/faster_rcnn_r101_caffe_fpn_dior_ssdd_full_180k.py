_base_ = "base.py"

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(checkpoint="open-mmlab://detectron2/resnet101_caffe"),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
    ),
)

# fold = 1
# percent = 1

# dataset_type = 'SHIPDataset'
# data_root = 'data/dior/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="HRSIDDataset",
        ann_file="data/dior_ssdd/annotations/instances_train2017.${fold}@${percent}_dior_annotations.json",
        img_prefix="data/dior_ssdd/images/",
    ),
    val=dict(
        type="HRSIDDataset",
        ann_file="data/ssdd/annotations/test.json",
        img_prefix="data/ssdd/JPEGImages/",
    ),
    test=dict(
        type="HRSIDDataset",
        ann_file="data/ssdd/annotations/test.json",
        img_prefix="data/ssdd/JPEGImages/",
    ),
)

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=2800)
checkpoint_config = dict(by_epoch=False, interval=200, max_keep_ckpts=10)
evaluation = dict(interval=200)

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="pre_release",
                name="${cfg_name}",
                config=dict(
                    fold="${fold}",
                    percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
