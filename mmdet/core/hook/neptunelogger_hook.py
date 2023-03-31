# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
from typing import Dict, Optional, Union

import mmcv
import mmcv.runner.hooks as mmvch
import numpy as np
from mmcv import Config
from mmcv.runner import HOOKS, EpochBasedRunner, IterBasedRunner
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook

from mmdet import version
from mmdet.core import DistEvalHook, EvalHook
from mmdet.core.visualization.image import imshow_det_bboxes


@HOOKS.register_module()
class NeptuneHook(mmvch.logger.neptune.NeptuneLoggerHook):
    """Logs training or evaluation metadata to neptune.ai.

    Args:
        init_kwargs (dict): A dictionary passed to the `neptune.init_run()`
            function to initialize and configure a Neptune run.
            Refer to https://docs.neptune.ai/api/neptune/#init_run
            for possible key-value pairs.
        api_token (str): Neptune API token.
            If None, the NEPTUNE_API_TOKEN environment variable is used.
        project (str): Neptune project name.
            If None, the NEPTUNE_PROJECT environment variable is used.
        interval (int): Checking interval (every k epochs/iterations).
            Default: 10
        base_namespace: Namespace where all the metadata logged by the hook
            is stored.
            Default: "training"
        log_model (bool): If the hook should save the final checkpoint.
            Requires CheckpointHook.
            Default: False
        log_checkpoint (bool): If the hook should log all the checkpoints
            generated by CheckpointHook.
            Requires CheckpointHook.
            Default: False
        num_eval_predictions (int): Maximum number of eval predictions to log.
            Requires EvalHook.
            Default: 50

    Example:

    ```
    log_config = dict(
        ...,
        hooks=[
            ...,
            dict(type='NeptuneHook',
                 init_kwargs={
                     'project': 'workspace-name/project-name',
                     'name': 'My MMDetection run',
                     'tags': ['mmdet', 'eval']
                 },
                 interval=20,
                 log_model=True,
                 num_eval_predictions=100)
        ])
    ```

    For more, see the Neptune docs:
    https://docs.neptune.ai/integrations/mmdetection/
    """

    INTEGRATION_VERSION_KEY = 'source_code/integrations/neptune-mmdetection'

    def __init__(self,
                 *,
                 init_kwargs: Optional[Dict] = None,
                 api_token: str = None,
                 project: str = None,
                 interval: int = 10,
                 base_namespace: str = 'training',
                 log_model: bool = False,
                 log_checkpoint: bool = False,
                 num_eval_predictions: int = 50,
                 **kwargs) -> None:
        super().__init__(interval=interval, **kwargs)

        try:
            import neptune
            import neptune.types as types
        except ImportError:
            raise ImportError('Neptune client library not installed.'
                              'Please refer to the installation guide:'
                              'https://docs.neptune.ai/setup/installation/')

        self.neptune = neptune
        self.types = types

        init_kwargs = init_kwargs if isinstance(init_kwargs, dict) else {}

        if 'api_token' in init_kwargs and api_token:
            if init_kwargs['api_token'] != api_token:
                raise ValueError('Two different api tokens were given!')
            api_token = init_kwargs.pop(api_token)

        if 'project' in init_kwargs and project:
            if init_kwargs['project'] != project:
                raise ValueError('Two different project names were given')
            project = init_kwargs.pop('project')

        self._run = neptune.init_run(
            api_token=api_token, project=project, **init_kwargs)
        self.base_namespace = base_namespace
        self.base_handler = self._run[base_namespace]

        self.log_model = log_model
        self.log_checkpoint = log_checkpoint

        self.num_eval_predictions = num_eval_predictions
        self.log_eval_predictions: bool = (num_eval_predictions > 0)

        self.ckpt_hook: Union[CheckpointHook, None] = None
        self.ckpt_interval: Union[int, None] = None

        self.eval_hook: Union[EvalHook, None] = None
        self.eval_interval: Union[int, None] = None

        self.val_dataset = None

        self.eval_image_indices = None

    def _log_integration_version(self) -> None:
        self._run[self.INTEGRATION_VERSION_KEY] = version.__version__

    def _log_config(self, runner) -> None:
        if runner.meta is not None and runner.meta.get('exp_name',
                                                       None) is not None:
            src_cfg_path = osp.join(runner.work_dir,
                                    runner.meta.get('exp_name', None))
            config = Config.fromfile(src_cfg_path)
            if osp.exists(src_cfg_path):
                self.base_handler['config'] = config.pretty_text
        else:
            runner.logger.warning('No meta information found in the runner. ')

    def _add_ground_truth(self, runner):
        # Get image loading pipeline
        from mmdet.datasets.pipelines import LoadImageFromFile
        img_loader = None
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, LoadImageFromFile):
                img_loader = t

        if img_loader is None:
            self.log_evaluation = False
            runner.logger.warning(
                'LoadImageFromFile is required to add images '
                'to the Neptune run.')
            return

        # Select the images to be logged.
        if not self.eval_image_indices:
            eval_image_indices = np.arange(len(self.val_dataset))
            # Set seed so that same validation set is logged each time.
            rng = np.random.default_rng(42)
            rng.shuffle(eval_image_indices)
            self.eval_image_indices = eval_image_indices[:self.
                                                         num_eval_predictions]

        CLASSES = self.val_dataset.CLASSES

        img_prefix = self.val_dataset.img_prefix
        for idx in self.eval_image_indices:
            img_info = self.val_dataset.data_infos[idx]
            image_name = img_info.get('filename', f'img_{idx}')
            # img_height, img_width = img_info['height'], img_info['width']

            img_meta = img_loader(
                dict(img_info=img_info, img_prefix=img_prefix))

            # Get image and convert from BGR to RGB
            image = mmcv.bgr2rgb(img_meta['img'])

            data_ann = self.val_dataset.get_ann_info(idx)
            bboxes = data_ann['bboxes']
            labels = data_ann['labels']
            # masks = data_ann.get('masks', None)

            # Get dict of bounding boxes to be logged.
            assert len(bboxes) == len(labels)
            im = imshow_det_bboxes(
                image, bboxes, labels, class_names=CLASSES, show=False)
            im = im / im.max()

            self._run[image_name].upload(self.types.File.as_image(im))

    @master_only
    def before_run(self, runner) -> None:
        """Logs config if exists, inspects the hooks in search of checkpointing
        and evaluation hooks.

        Raises a warning if checkpointing is enabled, but the dedicated hook is
        not present. Raises a warning if evaluation logging is enabled, but the
        dedicated hook is not present.
        """
        self._log_integration_version()
        self._log_config(runner)

        # Inspect CheckpointHook and EvalHook
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, (EvalHook, DistEvalHook)):
                self.eval_hook = hook

        if self.log_checkpoint:
            if self.ckpt_hook is None:
                self.log_checkpoint = False
                runner.logger.warning(
                    'To log checkpoint in NeptuneHook, `CheckpointHook` is'
                    'required, please check hooks in the runner.')
            else:
                self.ckpt_interval = self.ckpt_hook.interval

        if self.log_eval_predictions:
            if self.eval_hook is None:
                self.log_eval_predictions = False
                runner.logger.warning(
                    'To log evaluation with NeptuneHook, '
                    '`EvalHook` or `DistEvalHook` in mmdet '
                    'is required, please check whether the validation '
                    'is enabled.')
            else:
                self.eval_interval = self.eval_hook.interval
                self.val_dataset = self.eval_hook.dataloader.dataset

        self._add_ground_truth(runner)

    def _log_evaluation(self, runner, category):
        if self.eval_hook._should_evaluate(runner):

            results = self.eval_hook.latest_results

            eval_results = self.val_dataset.evaluate(
                results, logger='silent', **self.eval_hook.eval_kwargs)

            for key, value in eval_results.items():
                self.base_handler['val/' + category + '/' + key].append(value)

    def _log_buffer(self, runner, category) -> None:
        assert category in ['epoch', 'iter']
        # only record lr of the first param group
        cur_lr = runner.current_lr()
        self.base_handler['train/' + category + '/' +
                          'learning_rate'].extend(cur_lr)

        for key, value in runner.log_buffer.val_history.items():
            self.base_handler['train/' + category + '/' + key].append(
                value[-1])

    def _log_checkpoint(self,
                        runner,
                        ext='pth',
                        final=False,
                        mode='epoch') -> None:
        assert mode in ['epoch', 'iter']

        if self.ckpt_hook is None:
            return

        if final:
            file_name = 'final'
        else:
            file_name = f'{mode}_{getattr(runner, mode) + 1}'

        file_path = osp.join(self.ckpt_hook.out_dir, f'{file_name}.{ext}')

        neptune_checkpoint_path = osp.join('model/checkpoint/',
                                           f'{file_name}.{ext}')

        if not osp.exists(file_path):
            runner.logger.warning(
                f'Checkpoint {file_path} not found - skipping.')
            return
        with open(file_path, 'rb') as fp:
            self._run[neptune_checkpoint_path] = self.types.File.from_stream(
                fp)

    @master_only
    def after_train_iter(self, runner) -> None:
        """For an iter-based runner logs evaluation metadata, as well as
        checkpoints (if enabled by the user)."""
        if not isinstance(runner, IterBasedRunner):
            return

        log_eval = self.every_n_iters(runner, self.eval_hook.interval)
        if log_eval:
            self._log_evaluation(runner, 'iter')

        self._log_buffer(runner, 'iter')
        if self._should_upload_checkpoint(runner):
            self._log_checkpoint(runner, final=False, mode='iter')

    def _should_upload_checkpoint(self, runner) -> bool:
        if isinstance(runner, EpochBasedRunner):
            return self.log_checkpoint and \
                (runner.epoch + 1) % self.ckpt_interval == 0
        elif isinstance(runner, IterBasedRunner):
            return self.log_checkpoint and \
                (runner.iter + 1) % self.ckpt_interval == 0
        else:
            return False

    @master_only
    def after_train_epoch(self, runner) -> None:
        """For an epoch-based runner logs evaluation metadata, as well as
        checkpoints (if enabled by the user)."""
        if not isinstance(runner, EpochBasedRunner):
            return
        log_eval = self.every_n_epochs(runner, self.eval_hook.interval)
        if log_eval:
            self._log_evaluation(runner, 'epoch')

        self._log_buffer(runner, 'epoch')

        if self._should_upload_checkpoint(runner):
            self._log_checkpoint(runner, final=False)

    @master_only
    def after_run(self, runner) -> None:
        """If enabled by the user, logs final model checkpoint, syncs and stops
        the Neptune run."""
        if self.log_model:
            self._log_checkpoint(runner, final=True)

        runner.logger.info('Syncing with Neptune.ai')
        self._run.sync()
        self._run.stop()
