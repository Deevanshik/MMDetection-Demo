import argparse
from pytriton.decorators import batch
import numpy as np
import torch
import cv2
from mmdet.apis import inference_detector, init_detector
from mmdet.structures.det_data_sample import DetDataSample,SampleList
from typing import Optional
#You can replace those config and checkpoint with your own
config = './configs/swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth'
device = 'cuda:0'
device = torch.device(device)
model = init_detector(config, checkpoint, device=device)

# The function to be called by Triton, and it will be called for each batch.
# This function is for instance segmentation
@batch
def infer_func_instance_seg(**inputs):
    (input1_batch,) = inputs.values()
    print(type(input1_batch))
    print(input1_batch.shape)
    # Assuming input_array is a numpy array of shape (n, h, w, c)
    input1_batch = [input1_batch[i] for i in range(input1_batch.shape[0])]
    output1_batch_tensor = inference_detector(model,input1_batch) 
    # Calling the Python model inference
    print(len(output1_batch_tensor)) 
    print('inference completed')
    # Getting the output from the Python model inference
    output1 = []
    output2 = []
    output3 = []
    for output_tensor in output1_batch_tensor:
        output1_batch = output_tensor.pred_instances["masks"].cpu().detach().numpy()
        output2_batch = output_tensor.pred_instances["scores"].cpu().detach().numpy()
        output3_batch = output_tensor.pred_instances["bboxes"].cpu().detach().numpy()

        output1.append(output1_batch)
        output2.append(output2_batch)
        output3.append(output3_batch)
    output1_batch = np.array(output1)
    output2_batch = np.array(output2)
    output3_batch = np.array(output3)
    return {"masks":output1_batch,"scores":output2_batch,"bboxes":output3_batch}

# The function to be called by Triton, and it will be called for each batch.
# This function is for bbox detection
@batch
def infer_func_bbox_det(**inputs):
    (input1_batch,) = inputs.values()
    print(type(input1_batch))
    print(input1_batch.shape)
    # Assuming input_array is a numpy array of shape (n, h, w, c)
    input1_batch = [input1_batch[i] for i in range(input1_batch.shape[0])]
    output1_batch_tensor = inference_detector(model,input1_batch) 
    # Calling the Python model inference
    print(len(output1_batch_tensor)) 
    print('inference completed')
    # Getting the output from the Python model inference
    output1 = []
    output2 = []
    for output_tensor in output1_batch_tensor:
        output1_batch = output_tensor.pred_instances["scores"].cpu().detach().numpy()
        output2_batch = output_tensor.pred_instances["bboxes"].cpu().detach().numpy()
        output1.append(output1_batch)
        output2.append(output2_batch)
    output1_batch = np.array(output1)
    output2_batch = np.array(output2)
    return {"scores":output1_batch,"bboxes":output2_batch}


if __name__ == "__main__":
    from pytriton.model_config import ModelConfig, Tensor
    from pytriton.triton import Triton

    device = torch.device(args.device)

    model = init_detector(args.config, args.checkpoint, device=device)
    # Connecting inference callback with Triton Inference Server
    with Triton() as triton:
    # Load model into Triton Inference Server
        triton.bind(
        model_name="MMDetectionModel",
        infer_func=infer_func_instance_seg,#or replace with infer_func_bbox_det if there is no mask output
        inputs=[
        Tensor(dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
        Tensor(name="scores",dtype=np.float32, shape=(-1,)),
        Tensor(name="bboxes",dtype=np.float32, shape=(-1,)),
        Tensor(name="masks",dtype=np.bool_, shape=(-1,)),
        ],
        #outpus=[
        #   Tensor(name="scores",dtype=np.float32, shape=(-1,)),
        #  Tensor(name="bboxes",dtype=np.float32, shape=(-1,)),
        # ] if there is no mask output
        config=ModelConfig(max_batch_size=128)
        )
        triton.serve()

