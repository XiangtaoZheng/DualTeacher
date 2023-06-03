import argparse
import os
import os.path as osp
import time
import warnings

import numpy as np
from ssod.utils.ensemble_boxes import nms, weighted_boxes_fusion

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector

from ssod.utils import patch_config


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--show-dir", help="directory where painted images will be saved"
    )
    parser.add_argument(
        "--show-score-thr",
        type=float,
        default=0.3,
        help="score threshold (default: 0.3)",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both "
            "specified, --options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # cfg.model.pretrained = None
    if cfg.model.get("neck"):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get("rfp_backbone"):
                    if neck_cfg.rfp_backbone.get("pretrained"):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get("rfp_backbone"):
            if cfg.model.neck.rfp_backbone.get("pretrained"):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        cfg.work_dir = args.work_dir
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        json_file = osp.join(args.work_dir, f"eval_{timestamp}.json")
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    cfg = patch_config(cfg)
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    submodules = cfg.model.test_cfg.inference_on
    outputs = {}
    proposals = {}
    for submodule in submodules:
        cfg.model.test_cfg.inference_on = submodule
        if not distributed:
            modelx = MMDataParallel(model, device_ids=[0])
            outputs[submodule], proposals[submodule] = single_gpu_test(
                modelx, data_loader, args.show, args.show_dir, args.show_score_thr
            )
        else:
            modelx = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
            )
            outputs[submodule] = multi_gpu_test(modelx, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "type",
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            if 'teacher1' in outputs.keys() and 'teacher2' in outputs.keys():
                r1 = outputs['teacher1']
                r2 = outputs['teacher2']
                result = []
                for i in range(max(len(r1), len(r2))):
                    if len(r1[i][0]) == 0:
                        result.append(r2[i])
                        continue
                    if len(r2[i][0]) == 0:
                        result.append(r1[i])
                        continue
                    boxes = [r1[i][0][:, 0:4], r2[i][0][:, 0:4]]
                    scores = [r1[i][0][:, -1], r2[i][0][:, -1]]
                    label1 = [0] * len(r1[i][0])
                    label2 = [0] * len(r2[i][0])
                    labels = [label1, label2]
                    boxes_list, scores_list, _ = nms(boxes, scores, labels, iou_thr=0)
                    # boxes_list, scores_list, _ = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0)
                    result.append([np.hstack([boxes_list, np.expand_dims(scores_list, axis=1)])])
                outputs['teacher'] = result
            for key in outputs:
                metric = dataset.evaluate(outputs[key], **eval_kwargs)
                print(metric)
                metric_dict = dict(config=args.config, metric=metric)
                if args.work_dir is not None and rank == 0:
                    mmcv.dump(metric_dict, json_file)

    if 'teacher1' in proposals.keys() and 'teacher2' in proposals.keys():
        r1 = proposals['teacher1']
        r2 = proposals['teacher2']
        result = []
        for i in range(max(len(r1), len(r2))):
            result.append(torch.cat([r1[i], r2[i]], dim=0))
            # if len(r1[i]) == 0:
            #     result.append(r2[i])
            #     continue
            # if len(r2[i]) == 0:
            #     result.append(r1[i])
            #     continue
            # boxes = [np.array(r1[i][:, 0:4].cpu()), np.array(r2[i][:, 0:4].cpu())]
            # scores = [np.array(r1[i][:, -1].cpu()), np.array(r2[i][:, -1].cpu())]
            # label1 = [0] * len(r1[i])
            # label2 = [0] * len(r2[i])
            # labels = [label1, label2]
            # boxes_list, scores_list, _ = nms(boxes, scores, labels, iou_thr=0)
            # # boxes_list, scores_list, _ = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0)
            # result.append(torch.tensor(np.hstack([boxes_list, np.expand_dims(scores_list, axis=1)])))
        proposals['teacher'] = result
    outputs = {}
    for submodule in submodules:
        cfg.model.test_cfg.inference_on = submodule
        if not distributed:
            modelx = MMDataParallel(model, device_ids=[0])
            outputs[submodule], _ = single_gpu_test(
                modelx, data_loader, args.show, args.show_dir, args.show_score_thr, proposals['teacher']
            )
        else:
            modelx = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
            )
            outputs[submodule] = multi_gpu_test(modelx, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "type",
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            if 'teacher1' in outputs.keys() and 'teacher2' in outputs.keys():
                r1 = outputs['teacher1']
                r2 = outputs['teacher2']
                result = []
                for i in range(max(len(r1), len(r2))):
                    if len(r1[i][0]) == 0:
                        result.append(r2[i])
                        continue
                    if len(r2[i][0]) == 0:
                        result.append(r1[i])
                        continue
                    boxes = [r1[i][0][:, 0:4], r2[i][0][:, 0:4]]
                    scores = [r1[i][0][:, -1], r2[i][0][:, -1]]
                    label1 = [0] * len(r1[i][0])
                    label2 = [0] * len(r2[i][0])
                    labels = [label1, label2]
                    boxes_list, scores_list, _ = nms(boxes, scores, labels, iou_thr=0)
                    # boxes_list, scores_list, _ = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0)
                    result.append([np.hstack([boxes_list, np.expand_dims(scores_list, axis=1)])])
                outputs['teacher'] = result
            for key in outputs:
                metric = dataset.evaluate(outputs[key], **eval_kwargs)
                print(metric)
                metric_dict = dict(config=args.config, metric=metric)
                if args.work_dir is not None and rank == 0:
                    mmcv.dump(metric_dict, json_file)


if __name__ == "__main__":
    main()
