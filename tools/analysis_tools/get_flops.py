# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmdet.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--size-divisor',
        type=int,
        default=32,
        help='Pad the input image, the minimum size that is divisible '
        'by size_divisor, -1 means do not pad the image.')
    args = parser.parse_args()
    return args


def inference(args, logger):
    logger.warning(
        'if you want test all configs flops, please make sure torch>=1.12')
    # In torch<1.12, torch.jit don't support some configs compute flops
    # like configs/detectors, configs/yolact, so if you want to count the
    # flops mentioned above, you need update torch version. other configs
    # can compute flops in torch<1.12

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # TODO: The following usage is temporary and not safe
    # use hard code to convert mmSyncBN to SyncBN. This is a known
    # bug in mmengine, mmSyncBN requires a distributed environment，
    # this question involves models like configs/strong_baselines
    if hasattr(cfg, 'head_norm_cfg'):
        cfg['head_norm_cfg'] = dict(type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['bbox_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)
        cfg['model']['roi_head']['mask_head']['norm_cfg'] = dict(
            type='SyncBN', requires_grad=True)

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')
    divisor = args.size_divisor
    if divisor > 0:
        pad_h = int(np.ceil(h / divisor)) * divisor
        pad_w = int(np.ceil(w / divisor)) * divisor

    result = {}
    # Supports two ways to calculate flops,
    # 1. randomly generate a picture
    # 2. load a picture from the dataset
    # In two stage detectors, need batch_samples to compute flops
    # Only support use the second way.
    try:
        model = MODELS.build(cfg.model)
        result['ori_shape'] = (h, w)
        result['pad_shape'] = (pad_h, pad_w)
        input = torch.rand(1, 3, pad_h, pad_w)
        if torch.cuda.is_available():
            model.cuda()
            input = input.cuda()
        model = revert_sync_batchnorm(model)
        inputs = (input, )
        model.eval()
        outputs = get_model_complexity_info(
            model, None, inputs=inputs, show_table=False, show_arch=False)
        flops = outputs['flops']
        params = outputs['params']
        result['Get Types'] = 'direct: randomly generate a picture'

    except TypeError:
        logger.warning('Direct get flops failed, try to get flops with data')
        data_loader = Runner.build_dataloader(cfg.val_dataloader)
        data_batch = next(iter(data_loader))
        model = MODELS.build(cfg.model)
        if torch.cuda.is_available():
            model = model.cuda()
        model = revert_sync_batchnorm(model)
        model.eval()
        _forward = model.forward
        data = model.data_preprocessor(data_batch)
        ori_shape = data['data_samples'][0].ori_shape
        pad_shape = data['data_samples'][0].pad_shape
        result['ori_shape'] = ori_shape
        result['pad_shape'] = pad_shape

        del data_loader
        model.forward = partial(_forward, data_samples=data['data_samples'])
        outputs = get_model_complexity_info(
            model,
            None,
            inputs=data['inputs'],
            show_table=False,
            show_arch=False)
        flops = outputs['flops']
        params = outputs['params']
        result['Get Types'] = 'dataloader: load a picture from the dataset'

    flops = _format_size(flops)
    params = _format_size(params)
    result['flops'] = flops
    result['params'] = params

    return result


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    compute_type = result['Get Types']

    if args.size_divisor > 0 and \
            pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')
    print(f'{split_line}\nCompute type: {compute_type}\n'
          f'Input shape: {pad_shape}\nFlops: {flops}\n'
          f'Params: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')


if __name__ == '__main__':
    main()
