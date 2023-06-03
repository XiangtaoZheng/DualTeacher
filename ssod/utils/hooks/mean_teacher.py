from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
from ..logger import log_every_n


@HOOKS.register_module()
class MeanTeacher(Hook):
    def __init__(
        self,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        # assert hasattr(model, "teacher")
        # assert hasattr(model, "student")
        assert hasattr(model, "teacher") or (hasattr(model, "teacher1") and hasattr(model, "teacher2"))
        assert hasattr(model, "student") or (hasattr(model, "student1") and hasattr(model, "student2"))
        # only do it at initial stage
        if runner.iter == 0:
            log_every_n("Clone all parameters of student to teacher...")
            self.momentum_update(model, 0)

    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (curr_step + 1 + self.warm_up)
        )
        runner.log_buffer.output["ema_momentum"] = momentum
        self.momentum_update(model, momentum)

    def after_train_iter(self, runner):
        curr_step = runner.iter
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, curr_step
        )

    def momentum_update(self, model, momentum):
        if hasattr(model, "teacher") and hasattr(model, "student"):
            for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
                model.student.named_parameters(), model.teacher.named_parameters()
            ):
                tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
        if hasattr(model, "teacher1") and hasattr(model, "teacher2") \
            and hasattr(model, "student1") and hasattr(model, "student2"):
            for (src1_name, src1_parm), (src2_name, src2_parm), \
                (tgt1_name, tgt1_parm), (tgt2_name, tgt2_parm) in zip(
                model.student1.named_parameters(),
                model.student2.named_parameters(),
                model.teacher1.named_parameters(),
                model.teacher2.named_parameters(),
            ):
                tgt1_parm.data.mul_(momentum).add_(src1_parm.data, alpha=1 - momentum)
                tgt2_parm.data.mul_(momentum).add_(src2_parm.data, alpha=1 - momentum)
