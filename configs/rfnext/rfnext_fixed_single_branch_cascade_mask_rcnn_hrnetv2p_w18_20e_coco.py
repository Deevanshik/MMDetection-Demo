_base_ = '../hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py'
# model settings
model = dict(
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(18, 36)),
            stage3=dict(num_channels=(18, 36, 72)),
            stage4=dict(num_channels=(18, 36, 72, 144))),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w18')),
    neck=dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256),
    rfsearch_cfg=dict(
        logdir='./search_log/cascade_mask_rcnn_r2_101_fpn_20e_coco',
        mode='fixed_single_branch',
        rfstructure_file=  # noqa
        './configs/rfnext/search_log/cascade_mask_rcnn_r2_101_fpn_20e_coco/local_search_config_step11.json',  # noqa
        config=dict(
            search=dict(
                step=0,
                max_step=12,
                search_interval=1,
                exp_rate=0.5,
                init_alphas=0.01,
                normlize='absavg',
                mmin=1,
                mmax=24,
                S=2,
                finetune=False,
                skip_layer=['stem', 'layer1'])),
    ))

custom_hooks = [
    dict(
        type='RFSearch',
        logdir=model['rfsearch_cfg']['logdir'],
        config=model['rfsearch_cfg']['config'],
        mode=model['rfsearch_cfg']['mode'],
    ),
]
