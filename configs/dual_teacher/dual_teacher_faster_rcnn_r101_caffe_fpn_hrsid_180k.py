_base_ = "base.py"

# load_from = 'work_dirs/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k/10/1/iter_180000.pth'

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
    test_cfg=dict(inference_on=['teacher1', 'teacher2']),
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        sup1=dict(
            type="HRSIDDataset",
            ann_file="data/dior/dior_annotations.json",
            img_prefix="data/dior/images/",
        ),
        sup2=dict(
            type="HRSIDDataset",
            ann_file="data/hrsid/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="data/hrsid/images/",
        ),
        unsup=dict(
            type="HRSIDDataset",
            ann_file="data/hrsid/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="data/hrsid/images/",
        ),
    ),
    val=dict(
        type="HRSIDDataset",
        ann_file="data/hrsid/annotations/test2017.json",
        img_prefix="data/hrsid/images/",
    ),
    test=dict(
        type="HRSIDDataset",
        ann_file="data/hrsid/annotations/test2017.json",
        img_prefix="data/hrsid/images/",
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1, 1],
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
        load1_from='work_dirs/faster_rcnn_r101_caffe_fpn_dior_full_180k/100/0/iter_1000.pth',
        load2_from='work_dirs/faster_rcnn_r101_caffe_fpn_dior_hrsid_full_180k/${percent}/${fold}/iter_1000.pth',
    ),
    test_cfg=dict(inference_on='teacher'),
)

# fold = 1
# percent = 1

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=10)
evaluation = dict(interval=1000)

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
