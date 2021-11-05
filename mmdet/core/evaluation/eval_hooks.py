# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import bisect

import torch.distributed as dist
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(BaseEvalHook):
    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(EvalHook, self).__init__(*args, **kwargs)

        self.dynamic_intervals = dynamic_intervals
        if dynamic_intervals is not None:
            assert isinstance(dynamic_intervals, list)
            self.dynamic_steps = [0]
            self.dynamic_values = [self.interval]
            for dynamic_interval in dynamic_intervals:
                assert isinstance(dynamic_interval, tuple)
                self.dynamic_steps.append(dynamic_interval[0])
                self.dynamic_values.append(dynamic_interval[1])

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        if self.dynamic_intervals and self.by_epoch:
            step = bisect.bisect(self.dynamic_steps, (runner.epoch + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_values[step-1]

        if not (self.by_epoch and self.initial_flag):
            return
        if self.start is not None and runner.epoch >= self.start:
            self.after_train_epoch(runner)
        self.initial_flag = False

    def before_train_iter(self, runner):
        if self.dynamic_intervals and not self.by_epoch:
            step = bisect.bisect(self.dynamic_steps, (runner.iter + 1))
            self.interval = self.dynamic_values[step-1]

        """Evaluate the model only at the start of training by iteration."""
        if self.by_epoch or not self.initial_flag:
            return
        if self.start is not None and runner.iter >= self.start:
            self.after_train_iter(runner)
        self.initial_flag = False

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(EvalHook):

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 test_fn=None,
                 greater_keys=None,
                 less_keys=None,
                 broadcast_bn_buffer=True,
                 tmpdir=None,
                 gpu_collect=False,
                 out_dir=None,
                 file_client_args=None,
                 dynamic_intervals=None,
                 **eval_kwargs):

        if test_fn is None:
            from mmcv.engine import multi_gpu_test
            test_fn = multi_gpu_test

        super().__init__(
            dataloader,
            start=start,
            interval=interval,
            by_epoch=by_epoch,
            save_best=save_best,
            rule=rule,
            test_fn=test_fn,
            greater_keys=greater_keys,
            less_keys=less_keys,
            out_dir=out_dir,
            file_client_args=file_client_args,
            dynamic_intervals=dynamic_intervals,
            **eval_kwargs)

        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

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
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
