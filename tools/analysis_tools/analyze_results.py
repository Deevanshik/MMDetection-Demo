# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from multiprocessing import Pool

import numpy as np
from mmcv.transforms import to_tensor
from mmengine.config import Config, DictAction
<<<<<<< HEAD
<<<<<<< HEAD
from mmengine.fileio import load
from mmengine.utils import ProgressBar, check_file_exist, mkdir_or_exist
=======
>>>>>>> update
from mmengine.data import InstanceData
=======
from mmengine.data import InstanceData, PixelData
<<<<<<< HEAD
>>>>>>> Support panoptic_seg visualization
=======
from mmengine.fileio import load
>>>>>>> update
from mmengine.runner import Runner
from mmengine.utils import ProgressBar

from mmdet.datasets import get_loading_pipeline
from mmdet.evaluation import eval_map
from mmdet.registry import DATASETS, RUNNERS
from mmdet.structures import DetDataSample
from mmdet.utils import (register_all_modules, replace_cfg_vals,
                         update_data_root)
from mmdet.visualization import DetLocalVisualizer


def bbox_map_eval(det_result, annotation, nproc=4):
    """Evaluate mAP of single image det result.

    Args:
        det_result (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotation (dict): Ground truth annotations where keys of
             annotations are:

            - bboxes: numpy array of shape (n, 4)
            - labels: numpy array of shape (n, )
            - bboxes_ignore (optional): numpy array of shape (k, 4)
            - labels_ignore (optional): numpy array of shape (k, )

        nproc (int): Processes used for computing mAP.
            Default: 4.

    Returns:
        float: mAP
    """

    # use only bbox det result
    if isinstance(det_result, tuple):
        bbox_det_result = [det_result[0]]
    else:
        bbox_det_result = [det_result]
    # mAP
    iou_thrs = np.linspace(
        .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

    processes = []
    workers = Pool(processes=nproc)
    for thr in iou_thrs:
        p = workers.apply_async(eval_map, (bbox_det_result, [annotation]), {
            'iou_thr': thr,
            'logger': 'silent',
            'nproc': 1
        })
        processes.append(p)

    workers.close()
    workers.join()

    mean_aps = []
    for p in processes:
        mean_aps.append(p.get()[0])

    return sum(mean_aps) / len(mean_aps)


class ResultVisualizer:
    """Display and save evaluation results.

    Args:
        show (bool): Whether to show the image. Default: True.
        wait_time (float): Value of waitKey param. Default: 0.
        score_thr (float): Minimum score of bboxes to be shown.
           Default: 0.
        overlay_gt_pred (bool): Whether to plot gts and predictions on the
            same image. If False, predictions and gts will be plotted on two
            same image which will be concatenated in vertical direction.
            The image above is drawn with gt, and the image below is drawn
            with the prediction result. Default: False.
    """

    def __init__(self,
                 show=False,
                 wait_time=0,
                 score_thr=0,
                 overlay_gt_pred=False,
                 runner=None):
        self.show = show
        self.wait_time = wait_time
        self.score_thr = score_thr
        self.overlay_gt_pred = overlay_gt_pred
        self.visualizer = DetLocalVisualizer()
        self.runner = runner
        self.evaluator = runner.test_evaluator

    def _save_image_gts_results(self,
                                dataset,
                                results,
                                performances,
                                out_dir=None,
                                task='det'):
        """Display or save image with groung truths and predictions from a
        model.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection or panoptic segmentation
                results from test results pkl file.
            performances (dict): A dict contains samples's indices
                in dataset and model's performance on them.
            out_dir (str, optional): The filename to write the image.
                Defaults: None.
        """
        mkdir_or_exist(out_dir)

        for performance_info in performances:
            index, performance = performance_info
            data_info = dataset[index]
            data_info['gt_instances'] = data_info['instances']

            # calc save file path
            filename = data_info['img_path']
            fname, name = osp.splitext(osp.basename(filename))
            save_filename = fname + '_' + str(round(performance, 3)) + name
            out_file = osp.join(out_dir, save_filename)

            if task == 'det':
                gt_instances = InstanceData()
                gt_instances.bboxes = data_info['gt_bboxes']
                gt_instances.labels = data_info['gt_bboxes_labels']
                gt_samples = DetDataSample()
                gt_samples.gt_instances = gt_instances

                pred_instances = InstanceData()
                pred_instances.bboxes = results[index]['pred_instances'][
                    'bboxes']
                pred_instances.labels = results[index]['pred_instances'][
                    'labels']
                pred_instances.scores = results[index]['pred_instances'][
                    'scores']
                pred_samples = DetDataSample()
                pred_samples.pred_instances = pred_instances
            elif task == 'seg':
                gt_samples = DetDataSample()
                gt_sem_seg_data = dict(
                    sem_seg=to_tensor(data_info['gt_seg_map']))
                gt_samples.gt_sem_seg = PixelData(**gt_sem_seg_data)

                pred_samples = DetDataSample()
                pred_samples.pred_panoptic_seg = PixelData(
                    sem_seg=results[index]['pred_panoptic_seg']['sem_seg'])

            self.visualizer.add_datasample(
                'image',
                data_info['img'],
                gt_samples,
                pred_samples,
                show=self.show,
                pred_score_thr=self.score_thr,
                out_file=out_file)

    def evaluate_and_show(self,
                          dataset,
                          results,
                          topk=20,
                          show_dir='work_dir'):
        """Evaluate and show results.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection or panoptic segmentation
                results from test results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
            eval_fn (callable, optional): Eval function, Default: None.
        """

        self.visualizer.dataset_meta = dataset.metainfo

        assert topk > 0
        if (topk * 2) > len(dataset):
            topk = len(dataset) // 2

        good_dir = osp.abspath(osp.join(show_dir, 'good'))
        bad_dir = osp.abspath(osp.join(show_dir, 'bad'))

        if isinstance(results[0], dict):
            good_samples, bad_samples = self.panoptic_evaluate(
                dataset, results, topk=topk)
            self._save_image_gts_results(
                dataset, results, good_samples, good_dir, task='seg')
            self._save_image_gts_results(
                dataset, results, bad_samples, bad_dir, task='seg')
        elif isinstance(results[0], list):
            good_samples, bad_samples = self.detection_evaluate(
                dataset, results, topk=topk)
            self._save_image_gts_results(
                dataset, results, good_samples, good_dir, task='det')
            self._save_image_gts_results(
                dataset, results, bad_samples, bad_dir, task='det')
        else:
            raise 'The format of result is not supported yet. ' \
                'Current dict for panoptic segmentation and list ' \
                'for object detection are supported.'

    def detection_evaluate(self, dataset, results, topk=20, eval_fn=None):
        """Evaluation for object detection.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Object detection results from test
                results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.
            eval_fn (callable, optional): Eval function, Default: None.

        Returns:
            tuple: A tuple contains good samples and bad samples.
                good_mAPs (dict[int, float]): A dict contains good
                    samples's indices in dataset and model's
                    performance on them.
                bad_mAPs (dict[int, float]): A dict contains bad
                    samples's indices in dataset and model's
                    performance on them.
        """

        if eval_fn is None:
            eval_fn = bbox_map_eval
        else:
            assert callable(eval_fn)

        prog_bar = ProgressBar(len(results))
        _mAPs = {}
        data_info = {}
        for i, (result, ) in enumerate(zip(results)):

            # self.dataset[i] should not call directly
            # because there is a risk of mismatch
            data_info = dataset.prepare_data(i)
            data_info['bboxes'] = data_info['gt_bboxes']
            data_info['labels'] = data_info['gt_bboxes_labels']

            pred = result['pred_instances']
            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            dets = []
            for label in range(len(dataset.metainfo['CLASSES'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)
            mAP = eval_fn(dets, data_info)

            _mAPs[i] = mAP
            prog_bar.update()
        # descending select topk image
        _mAPs = list(sorted(_mAPs.items(), key=lambda kv: kv[1]))
        good_mAPs = _mAPs[-topk:]
        bad_mAPs = _mAPs[:topk]

        return good_mAPs, bad_mAPs

    def panoptic_evaluate(self, dataset, results, topk=20):
        """Evaluation for panoptic segmentation.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Panoptic segmentation results from test
                results pkl file.
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20.

        Returns:
            tuple: A tuple contains good samples and bad samples.
                good_pqs (dict[int, float]): A dict contains good
                    samples's indices in dataset and model's
                    performance on them.
                bad_pqs (dict[int, float]): A dict contains bad
                    samples's indices in dataset and model's
                    performance on them.
        """
        # image to annotations
        gt_json = dataset.coco.img_ann_map

<<<<<<< HEAD
        result_files, tmp_dir = dataset.format_results(results)
        pred_json = load(result_files['panoptic'])['annotations']
        pred_folder = osp.join(tmp_dir.name, 'panoptic')
        gt_folder = dataset.seg_prefix
=======
        pqs = {}
        prog_bar = ProgressBar(len(results))
>>>>>>> update

        pqs = {}
        prog_bar = ProgressBar(len(results))
        for i in range(len(results)):
            data_sample = {}
            for k in dataset[i].keys():
                data_sample[k] = dataset[i][k]
            data_batch = {}
            data_batch['data_sample'] = data_sample

            self.evaluator.process([data_batch], [results[i]])
            metrics = self.evaluator.evaluate(1)

            pqs[i] = metrics['coco_panoptic/PQ']
            prog_bar.update()

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()

        # descending select topk image
        pqs = list(sorted(pqs.items(), key=lambda kv: kv[1]))
        good_pqs = pqs[-topk:]
        bad_pqs = pqs[:topk]

        return good_pqs, bad_pqs


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        'show_dir', help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=0,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--topk',
        default=20,
        type=int,
        help='saved Number of the highest topk '
        'and lowest topk after index sorting')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0,
        help='score threshold (default: 0.)')
    parser.add_argument(
        '--overlay-gt-pred',
        action='store_true',
        help='whether to plot gts and predictions on the same image.'
        'If False, predictions and gts will be plotted on two same'
        'image which will be concatenated in vertical direction.'
        'The image above is drawn with gt, and the image below is'
        'drawn with the prediction result.')
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
    args = parser.parse_args()
    return args


def main():
    register_all_modules()
    args = parse_args()

    check_file_exist(args.prediction_path)

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.test_dataloader.dataset.test_mode = True

    cfg.test_dataloader.pop('batch_size', 0)
    if cfg.train_dataloader.dataset.type in ('MultiImageMixDataset',
                                             'ClassBalancedDataset',
                                             'RepeatDataset', 'ConcatDataset'):
        cfg.test_dataloader.dataset.pipeline = get_loading_pipeline(
            cfg.train_dataloader.dataset.dataset.pipeline)
    else:
        cfg.test_dataloader.dataset.pipeline = get_loading_pipeline(
            cfg.train_dataloader.dataset.pipeline)

<<<<<<< HEAD
    dataset = build_dataset(cfg.data.test)
=======
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
>>>>>>> update
    outputs = load(args.prediction_path)

    cfg.work_dir = args.show_dir
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    result_visualizer = ResultVisualizer(args.show, args.wait_time,
                                         args.show_score_thr,
                                         args.overlay_gt_pred, runner)
    result_visualizer.evaluate_and_show(
        dataset, outputs, topk=args.topk, show_dir=args.show_dir)


if __name__ == '__main__':
    main()
