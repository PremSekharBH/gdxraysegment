import os
import cv2
import json
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt

# Step 1: Setup Detectron2 model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the detection threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Step 2: Function to convert inference results to COCO format
def convert_to_coco_format(instances, image_id):
    annotations = []
    for i in range(len(instances)):
        annotation = {}
        annotation["image_id"] = image_id
        annotation["category_id"] = int(instances.pred_classes[i])
        annotation["bbox"] = instances.pred_boxes[i].tensor.numpy().tolist()[0]
        annotation["score"] = float(instances.scores[i])

        # Convert binary mask to RLE format
        rle = mask_util.encode(np.array(instances.pred_masks[i].numpy(), dtype=np.uint8, order="F"))
        rle["counts"] = rle["counts"].decode("utf-8")  # For JSON compatibility
        annotation["segmentation"] = rle
        annotation["area"] = mask_util.area(rle).item()

        annotations.append(annotation)
    return annotations

# Step 3: Perform inference on all images in a folder
image_folder = "path_to_your_image_folder"
output_json_path = "model_inference_results.json"
coco_results = []

for image_filename in os.listdir(image_folder):
    if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
        image_path = os.path.join(image_folder, image_filename)
        image = cv2.imread(image_path)
        outputs = predictor(image)
        
        image_id = int(os.path.splitext(image_filename)[0])  # Assuming image filename is numeric
        coco_results.extend(convert_to_coco_format(outputs["instances"].to("cpu"), image_id))

# Step 4: Save results to a COCO format JSON file
with open(output_json_path, "w") as f:
    json.dump(coco_results, f)

# Step 5: Compare Ground Truth with Model Inference and Generate PR Curve
ground_truth_json = "path_to_ground_truth_annotations.json"  # Provide the path to your ground truth JSON file

# Load Ground Truth and Model Inference JSON files
coco_gt = COCO(ground_truth_json)
coco_dt = coco_gt.loadRes(output_json_path)

# Initialize COCOeval object
coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")

# Evaluate on different score thresholds
coco_eval.params.iouThrs = [0.5]
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Extract precision-recall curve data
precisions = coco_eval.eval['precision']
recalls = coco_eval.params.recThrs

# Plot PR Curve
plt.figure(figsize=(8, 6))
for i, p in enumerate(precisions):
    plt.plot(recalls, p.mean(axis=(0, 1)), label=f'IoU={coco_eval.params.iouThrs[i]:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
