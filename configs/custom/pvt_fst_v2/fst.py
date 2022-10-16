_base_ = 'fstb0.py'
model = dict(
    backbone=dict(
        embed_dims=64,
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b1.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))
