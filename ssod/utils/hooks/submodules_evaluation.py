import os.path as osp

import numpy as np
from ssod.utils.ensemble_boxes import nms

import torch.distributed as dist
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, LoggerHook, WandbLoggerHook
from mmdet.core import DistEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


@HOOKS.register_module()
class SubModulesDistEvalHook(DistEvalHook):
    def __init__(self, *args, evaluated_modules=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluated_modules = evaluated_modules

    def before_run(self, runner):
        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        assert hasattr(model, "submodules")
        assert hasattr(model, "inference_on")

    def after_train_iter(self, runner):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch and self._should_evaluate(runner):
            for hook in runner._hooks:
                if isinstance(hook, WandbLoggerHook):
                    _commit_state = hook.commit
                    hook.commit = False
                if isinstance(hook, LoggerHook):
                    hook.after_train_iter(runner)
                if isinstance(hook, WandbLoggerHook):
                    hook.commit = _commit_state
            runner.log_buffer.clear()

            self._do_evaluate(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.

        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, ".eval_hook")

        if is_module_wrapper(runner.model):
            model_ref = runner.model.module
        else:
            model_ref = runner.model
        if not self.evaluated_modules:
            submodules = model_ref.submodules
        else:
            submodules = self.evaluated_modules
        key_scores = []
        from mmdet.apis import multi_gpu_test

        results = {}
        for submodule in submodules:
            # change inference on
            model_ref.inference_on = submodule
            results[submodule] = multi_gpu_test(
                runner.model,
                self.dataloader,
                tmpdir=tmpdir,
                gpu_collect=self.gpu_collect,
            )
            if runner.rank == 0:
                key_score = self.evaluate(runner, results[submodule], prefix=submodule)
                if key_score is not None:
                    key_scores.append(key_score)
        # if 'teacher1' in results.keys() and 'teacher2' in results.keys():
        #     r1 = results['teacher1']
        #     r2 = results['teacher2']
        #     result = []
        #     for i in range(max(len(r1), len(r2))):
        #         if len(r1[i][0]) == 0:
        #             result.append(r2[i])
        #             continue
        #         if len(r2[i][0]) == 0:
        #             result.append(r1[i])
        #             continue
        #         boxes = [r1[i][0][:, 0:4], r2[i][0][:, 0:4]]
        #         scores = [r1[i][0][:, -1], r2[i][0][:, -1]]
        #         label1 = [0] * len(r1[i][0])
        #         label2 = [0] * len(r2[i][0])
        #         labels = [label1, label2]
        #         boxes_list, scores_list, _ = nms(boxes, scores, labels, iou_thr=0)
        #         result.append([np.hstack([boxes_list, np.expand_dims(scores_list, axis=1)])])
        #     key_score = self.evaluate(runner, result, prefix='teacher')

        if runner.rank == 0:
            runner.log_buffer.ready = True
            if len(key_scores) == 0:
                key_scores = [None]
            best_score = key_scores[0]
            for key_score in key_scores:
                if hasattr(self, "compare_func") and self.compare_func(
                    key_score, best_score
                ):
                    best_score = key_score

            print("\n")
            # runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
            if self.save_best:
                self._save_ckpt(runner, best_score)

    def evaluate(self, runner, results, prefix=""):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs
        )
        for name, val in eval_res.items():
            runner.log_buffer.output[(".").join([prefix, name])] = val

        if self.save_best is not None:
            if self.key_indicator == "auto":
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None
