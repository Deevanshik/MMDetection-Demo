_base_ = '../mask_rcnn/mask-rcnn_r50-fpn_1x_coco.py'
model = dict(
    data_preprocessor=dict(pad_size_divisor=64),
    neck=dict(
        type='FPN_CARAFE',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        start_level=0,
        end_level=-1,
        norm_cfg=None,
        act_cfg=None,
        order=('conv', 'norm', 'act'),
        upsample_cfg=dict(
            type='carafe',
            up_kernel=5,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64)),
    roi_head=dict(
        mask_head=dict(
            upsample_cfg=dict(
                type='carafe',
                scale_factor=2,
                up_kernel=5,
                up_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64))))
