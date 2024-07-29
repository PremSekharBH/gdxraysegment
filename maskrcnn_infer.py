import os
import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
import matplotlib.pyplot as plt
import time




def draw_masks_on_image(image, masks, color=(0, 255, 0), thickness=2):
    image_with_contours = image.copy()
    for mask in masks:
        mask_binary = np.uint8(mask > 0.5)
        mask_resized = cv2.resize(mask_binary, (image.shape[1], image.shape[0]))
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_with_contours, contours, -1, color, thickness)
    return image_with_contours

import json

#create image into smaller chunks and detect defects in those regions 
def process_image(image, parts=4):
    h, w = image.shape[:2]
    #print("parts value", parts)
    part_height = h // parts
    part_width = w // parts
    
    image_parts = []
    draw_image = np.zeros((image.shape), np.uint8)
    #print("DRAW image shape", draw_image.shape, image.shape)
    for i in range(parts):
        for j in range(parts):
            part = image[i*part_height:(i+1)*part_height, j*part_width:(j+1)*part_width]
            #print(part.shape, part_height, part_width)
            image_parts.append(part)
            draw_part = predict_output(part)
            draw_image[i*part_height:(i+1)*part_height, j*part_width:(j+1)*part_width] = draw_part
    # plt.imshow(draw_image)
    # plt.show()
    return draw_image

def predict_output(image):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    # with open(model_config_yml,'r') as file:
    #     self.model_config = yaml.safe_load(file)
    cfg.merge_from_file(r"C:\work\welding\train\cimc_tiff_train1\org_model-config.yaml")
    cfg.MODEL.WEIGHTS = r"C:\work\welding\train\cimc_tiff_train1\model_best.pth"  # Replace with your model weights file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for detections

    # Create a predictor
    predictor = DefaultPredictor(cfg)
    start_time = time.time()
    outputs = predictor(image)

    # Get predictions
    instances = outputs["instances"]

    # Convert predictions to numpy arrays
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()
    masks = instances.pred_masks.cpu().numpy()

    result_image = draw_masks_on_image(image, masks)
    print("Inference Done for image part")
    print("--- %s seconds ---" % (time.time() - start_time))
    return result_image

def inference_on_image(image_path):
    inference_start_time = time.time()
    image =cv2.imread(image_path)
    out_image = process_image(image,4)
    print("Time taken for inference", time.time() - inference_start_time)
    return out_image                                      

