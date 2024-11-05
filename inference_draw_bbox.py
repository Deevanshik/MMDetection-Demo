from mmdet.apis import init_detector, inference_detector
import cv2

metainfo = {
    'classes': ['Dolphin', 'Elephant', 'Guitar', 'Piano', 'Rabbit', 'Violin']
}
attribute_metainfo = {
    'type': ['Animal', 'Instrument']
}
# Paths to config and checkpoint
config_file = '/home/g-zhu/mmdetection_test/test.mmdetection/rtmdet_tiny_customized_animal_instrument.py'
checkpoint_file = '/home/g-zhu/sora/mmdetection_abate/mmdetection/work_dirs/rtmdet_l_8xb32-300e_coco_test/epoch_100.pth'
img_path = '/home/g-zhu/sora/mmdetection_abate/mmdetection/coco_customized/coco_validation/data/00714f9e7d062900.jpg'
# config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
# checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
# img_path = '/home/g-zhu/sora/mmdetection_abate/mmdetection/demo/demo.jpg'

img = cv2.imread(img_path)
resize_img = cv2.resize(img, (640, 640)) 

model = init_detector(config_file, checkpoint_file, device='cpu')

result = inference_detector(model, resize_img)

bboxes = result.pred_instances.bboxes.cpu().numpy()
scores = result.pred_instances.scores.cpu().numpy()
labels = result.pred_instances.labels.cpu().numpy()
attribute_labels = result.pred_instances.attribute_labels.cpu().numpy()
print(scores)
print(labels)
# Threshold for filtering detections
score_threshold = 0.3

# Iterate over detections
for i in range(len(scores)):
    if scores[i] >= score_threshold:
        # Get bounding box coordinates and class labels
        bbox = bboxes[i].astype(int)
        label = labels[i]
        attribute_label = attribute_labels[i]
        
        # Retrieve class name and attribute name
        class_name = metainfo['classes'][label]
        attribute_name = attribute_metainfo['type'][attribute_label]

        # Draw bounding box on the image
        cv2.rectangle(resize_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        
        # Annotate with class and attribute names
        text = f"{class_name} ({attribute_name})"
        cv2.putText(resize_img, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display or save the annotated image
cv2.imwrite('Annotated_Image.jpg', resize_img)