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
import multiprocessing as mp

def initialize_predictor(config_path, weights_path, score_thresh=0.5):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    return DefaultPredictor(cfg)

def draw_masks_on_image(image, masks, color=(0, 255, 0), thickness=2):
    image_with_contours = image.copy()
    for mask in masks:
        mask_binary = np.uint8(mask > 0.5)
        mask_resized = cv2.resize(mask_binary, (image.shape[1], image.shape[0]))
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_with_contours, contours, -1, color, thickness)
    return image_with_contours

def predict_output(image_chunk, predictor):
    outputs = predictor(image_chunk)
    instances = outputs["instances"]
    masks = instances.pred_masks.cpu().numpy()
    result_image = draw_masks_on_image(image_chunk, masks)
    return result_image

def process_chunk(args):
    image_chunk, predictor = args
    return predict_output(image_chunk, predictor)

def process_image(image, predictor, parts=4):
    h, w = image.shape[:2]
    part_height = h // parts
    part_width = w // parts

    image_parts = []
    coords = []

    for i in range(parts):
        for j in range(parts):
            part = image[i*part_height:(i+1)*part_height, j*part_width:(j+1)*part_width]
            image_parts.append(part)
            coords.append((i, j))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        result_parts = pool.map(process_chunk, [(part, predictor) for part in image_parts])

    draw_image = np.zeros(image.shape, np.uint8)
    for (i, j), part in zip(coords, result_parts):
        draw_image[i*part_height:(i+1)*part_height, j*part_width:(j+1)*part_width] = part

    return draw_image

def inference_on_image(image_path, config_path, weights_path):
    inference_start_time = time.time()
    image = cv2.imread(image_path)
    predictor = initialize_predictor(config_path, weights_path)
    out_image = process_image(image, predictor, parts=4)
    print("Time taken for inference", time.time() - inference_start_time)
    return out_image

if __name__ == "__main__":
    image_path = "path_to_your_image"
    config_path = r"C:\work\welding\train\cimc_tiff_train1\org_model-config.yaml"
    weights_path = r"C:\work\welding\train\cimc_tiff_train1\model_best.pth"
    result_image = inference_on_image(image_path, config_path, weights_path)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.show()
