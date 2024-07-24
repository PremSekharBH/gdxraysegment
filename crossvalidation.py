#add cross-validation modifications here
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo
import os
import torch
import json
import numpy as np
from sklearn.model_selection import KFold
from fvcore.common.checkpoint import Checkpointer
from detectron2.engine import BestCheckpointer
from PIL import ImageFile
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog

ImageFile.LOAD_TRUNCATED_IMAGES = True

def save_cfg(cfg, output_path):
    with open(output_path, 'w') as f:
        f.write(cfg.dump())

# ----------------------------------
train_json_path = "C:\\work\\CIMC_Data\\COCO_CIMC_RR\\annotations\\instances_train2017.json"
train_images_path = "C:\\work\\CIMC_Data\\COCO_CIMC_RR\\train2017"
total_iteration = 6000
eval_period = 500
device = 'cuda:0'
output_dir = 'cimc_tiff_train1'
num_folds = 5
# ----------------------------------

# Load COCO annotations
with open(train_json_path) as f:
    coco_data = json.load(f)

images = coco_data['images']
annotations = coco_data['annotations']

# Create a mapping from image ID to annotations
image_to_annotations = {img['id']: [] for img in images}
for annotation in annotations:
    image_to_annotations[annotation['image_id']].append(annotation)

# Create an array of image IDs
image_ids = np.array([img['id'] for img in images])

# Define the number of folds
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Create train-test splits
splits = list(kf.split(image_ids))

def create_fold_dataset(fold, train_idx, val_idx, images, image_to_annotations):
    train_images = [images[i] for i in train_idx]
    val_images = [images[i] for i in val_idx]

    train_annotations = [ann for img in train_images for ann in image_to_annotations[img['id']]]
    val_annotations = [ann for img in val_images for ann in image_to_annotations[img['id']]]

    train_dataset = {
        "images": train_images,
        "annotations": train_annotations,
    }

    val_dataset = {
        "images": val_images,
        "annotations": val_annotations,
    }

    return train_dataset, val_dataset

class ImgaugMapper:
    def __init__(self):
        pass

    def __call__(self, dataset_dict):
        image_aug = utils.read_image(dataset_dict["file_name"], format="BGR")
        new_annos = []
        for obj in dataset_dict["annotations"]:
            x, y, w, h = obj["bbox"]
            rle = obj["segmentation"]
            new_annos.append({"bbox_mode": BoxMode.XYXY_ABS,
                              "category_id": int(obj['category_id']),
                              "bbox": [np.float64(x), np.float64(y), np.float64(x + w), np.float64(y + h)],
                              'segmentation': rle
                              })
        dataset_dict["image"] = torch.as_tensor(image_aug.transpose(2, 0, 1).copy())
        dataset_dict["instances"] = utils.annotations_to_instances(new_annos, image_aug.shape[:2])
        return dataset_dict

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(cfg, mapper=ImgaugMapper(), sampler=sampler)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(os.path.join(cfg.OUTPUT_DIR, "coco_eval"), exist_ok=True)
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        bestcheckpointer = Checkpointer(model=self.model, save_dir=self.cfg.OUTPUT_DIR)
        hooks.insert(-1, BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, bestcheckpointer, "segm/AP", mode='max'))
        return hooks

results = []

for fold, (train_idx, val_idx) in enumerate(splits):
    print(f"Training fold {fold + 1}/{num_folds}")
    
    train_dataset, val_dataset = create_fold_dataset(fold, train_idx, val_idx, images, image_to_annotations)
    
    train_ds_name = f"train_fold_{fold}"
    val_ds_name = f"val_fold_{fold}"
    
    DatasetCatalog.register(train_ds_name, lambda d=train_dataset: d)
    MetadataCatalog.get(train_ds_name).set(thing_classes=["defect"])
    
    DatasetCatalog.register(val_ds_name, lambda d=val_dataset: d)
    MetadataCatalog.get(val_ds_name).set(thing_classes=["defect"])
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_ds_name,)
    cfg.DATASETS.TEST = (val_ds_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0015
    cfg.SOLVER.MAX_ITER = total_iteration
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = device
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.OUTPUT_DIR = os.path.join(output_dir, f"fold_{fold}")
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(cfg.OUTPUT_DIR, "model-config.yaml"), "w") as f:
        f.write(cfg.dump())
    
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # Evaluate the model on the validation set
    evaluator = COCOEvaluator(val_ds_name, cfg, False, output_folder=os.path.join(cfg.OUTPUT_DIR, "coco_eval"))
    val_loader = build_detection_test_loader(cfg, val_ds_name)
    metrics = trainer.test(cfg, trainer.model, evaluators=[evaluator])
    results.append(metrics)

# Save the cross-validation results
with open(os.path.join(output_dir, "cross_validation_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print("Cross-validation results:", results)

