# Epoch(val) [12][548]   bbox_mAP: 0.2670, bbox_mAP_50: 0.5150, bbox_mAP_75: 0.2500, bbox_mAP_s: 0.1830, bbox_mAP_m: 0.3810, bbox_mAP_l: 0.4370,
# bbox_mAP_copypaste: 0.267 0.515 0.250 0.183 0.381 0.437
export GPU=1 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_1x_VisDrone640"
tools/dist_train.sh configs2/VisDrone/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=16

# Epoch(val) [12][137]   bbox_mAP: 0.2600, bbox_mAP_50: 0.5040, bbox_mAP_75: 0.2400, bbox_mAP_s: 0.1770, bbox_mAP_m: 0.3750, bbox_mAP_l: 0.4210,
# bbox_mAP_copypaste: 0.260 0.504 0.240 0.177 0.375 0.421
export GPU=4 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_noaem_noffm_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# Epoch(val) [12][137]   bbox_mAP: 0.2610, bbox_mAP_50: 0.5050, bbox_mAP_75: 0.2430, bbox_mAP_s: 0.1770, bbox_mAP_m: 0.3740, bbox_mAP_l: 0.4030,
# bbox_mAP_copypaste: 0.261 0.505 0.243 0.177 0.374 0.403
export GPU=4 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_noffm_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# Epoch(val) [12][137]   bbox_mAP: 0.1140, bbox_mAP_50: 0.2680, bbox_mAP_75: 0.0830, bbox_mAP_s: 0.0810, bbox_mAP_m: 0.1550, bbox_mAP_l: 0.1610,
# bbox_mAP_copypaste: 0.114 0.268 0.083 0.081 0.155 0.161
# 看起来还算正常，难道是因为数据集太小的原因？
export GPU=4 && LR=0.16 && CONFIG="faster_rcnn_r18_lka_fpn_1x_VisDrone640"
tools/dist_train.sh configs2/VisDrone/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=16

#  Epoch(val) [12][137]   bbox_mAP: 0.1030, bbox_mAP_50: 0.2490, bbox_mAP_75: 0.0730, bbox_mAP_s: 0.0750, bbox_mAP_m: 0.1400, bbox_mAP_l: 0.1450,
#  bbox_mAP_copypaste: 0.103 0.249 0.073 0.075 0.140 0.145
#  看起来不用额外调参了
export GPU=4 && LR=0.08 && CONFIG="faster_rcnn_r18_lka_fpn_1x_VisDrone640"
tools/dist_train.sh configs2/VisDrone/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=16


