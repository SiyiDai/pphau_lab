# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from functools import partial
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import torch, torchvision
import numpy as np
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import pycocotools.mask as coco_mask
from detectron2.config import get_cfg
import os

# Visualizer packages
from matplotlib import pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


# Initialize the configuration of the model output to yaml file
# Before using this function, make sure that the dataset is already
# successfully registered in class DatasetCatalog, otherwise could 
# cause error.
# All parameters in this cfg initialization function can be modified
# to fit the custom senario

def Init_Save_Cfg(**kwargs):
    # initialize some deflaut paths in case that there
    # is no argument given
    default_cfg_path = "assets/mask_rcnn/cfg_mivos.yaml"
    default_model_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"    
    default_num_categories = 2
    cfg = get_cfg()
    
    if 'cfg_path' in kwargs.keys():
        cfg.merge_from_file(kwargs['cfg_path'])
    else:
        cfg.merge_from_file(default_cfg_path) 
    #Train/test set assertion
    cfg.DATASETS.TRAIN = ("Mivos",)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.DEVICE= 'cuda:0'   #for GPU, otherwise 'cpu'
    # initialize the model weights with custom weights or pretrained weights
    if 'model_path' in kwargs.keys():
        cfg.MODEL.WEIGHTS = kwargs['model_path']
    else:
        cfg.MODEL.WEIGHTS = default_model_path
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 300 # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset
    if 'num_categories' in kwargs.keys():
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = kwargs['num_categories']
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = default_num_categories
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    
    # save configuration unter current folder
    with open('cfghw2_2.yaml', 'w') as f:
        f.write(cfg.dump())
    
    # if you want to merge cfg from other cfg without reading os path, the output of
    # this function can also be used
    return cfg
        
# Visualizer initialization, argument is the class Detectron2Wrapper

def init_visualizer(det, test_path = 'mask_rcnn/Mivos_cus/images/frame0005.jpg'):    
    img_test = cv2.imread(test_path)
    
    result = det.predictor(img_test)
    
    v = Visualizer(img_test[:, :, ::-1],
                       metadata=det.meta, 
                       scale=0.8, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
    
    v = v.draw_instance_predictions(result["instances"].to("cpu"))
    
    plt.figure()
    
    plt.imshow(v.get_image()[:, :, ::-1])
        
# Create object masks

def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}

    # 'r': 1.0, 'g': 0.9215686321258545, 'b': 0.01568627543747425, 'a': 1.0

    # sub_masks[1] = mask_image[:,:,1.0, 0.9215686321258545, 0.01568627543747425]
    # sub_masks[1] = mask_image[:,:]
    img1 = np.zeros((width, height), dtype=np.uint8)
    img2 = np.zeros((width, height), dtype=np.uint8)
    img3 = np.zeros((width, height), dtype=np.uint8)
    img4 = np.zeros((width, height), dtype=np.uint8)
    img5 = np.zeros((width, height), dtype=np.uint8)
    img6 = np.zeros((width, height), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            pixel = mask_image[x, y]
            if pixel[2] == 0 and pixel[1] == (int)(128) and pixel[0] == (int)(128):
                img1[x, y] = 255
            elif pixel[2] == (int)(128) and pixel[1] == (int)(0) and pixel[0] == (int)(0):
                img2[x, y] = 255
            elif pixel[2] == (int)(128) and pixel[1] == (int)(128) and pixel[0] == (int)(0):
                img3[x, y] = 255
            elif pixel[2] == (int)(0) and pixel[1] == (int)(128) and pixel[0] == (int)(0):
                img4[x, y] = 255
            elif pixel[2] == (int)(0) and pixel[1] == (int)(0) and pixel[0] == (int)(128):
                img5[x, y] = 255
            elif pixel[2] == (int)(128) and pixel[1] == (int)(0) and pixel[0] == (int)(128):
                img6[x, y] = 255
            # print(pixel)

    sub_masks[0] = img1  # person
    sub_masks[1] = img2  # chekerboard
    sub_masks[2] = img3  # spoon
    sub_masks[3] = img4  # milk
    sub_masks[4] = img5  # bowl
    sub_masks[5] = img6  # oats
    return sub_masks

# dex-ycb hand segmentation function

def dexycb_hand_seg_func_mivos(num_samples=-1, ignore_background=False, data_dir='/home/kolz14w/haucode/hw2/assets/mask_rcnn/content', step_size=50):
    subjects = [f for f in os.listdir(data_dir) if 'subject' in f]
    print('Subjects: ',subjects)
    lst = []
    
    prefix = f"{data_dir}"
    length = num_samples
    print(length)
    for i in range(0, length, 1):
        color_file = f"{prefix}/images/frame%04d.jpg" % (i * step_size)
        seg_mask = f"{prefix}/mask/%05d.png" % (i * step_size)
        print(color_file)
        print(seg_mask)

        assert os.path.exists(color_file)
        assert os.path.exists(seg_mask)
        seg_img = cv2.imread(seg_mask)
        height, width, channels = seg_img.shape

        a = create_sub_masks(seg_img, height, width)
        annotations = []

        for j in range(len(a)):
            rel_mask = coco_mask.encode(np.asfortranarray(a[j]))
            rows, cols = np.where(a[j])
            category_label, category_id = get_category2(j)
            print(rows.any())

            if rows.any():
                print("Entered Annotator")
                annotation = {
                    'bbox': [min(cols), min(rows), max(cols), max(rows)],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': category_id,
                    'segmentation': rel_mask
                }

                print(annotation)

                annotations.append(annotation)

        dct = {
            'file_name': color_file,
            'height': height,
            'width': width,
            'image_id': i,
            'annotations': annotations,
        }

        lst.append(dct)
        if len(lst) == length:
            return lst

# Get object class

def get_category2(id):
    
    k = OBJ_CLASSES[id]

    return k, INV_OBJ_CATEGORIES[k]

# Dataset registeration

def data_reg(dataset_name, dataset_path):
    DatasetCatalog.clear()
    DatasetCatalog.register(dataset_name, partial(dexycb_hand_seg_func_mivos,
                                                 num_samples = 5,
                                                 ignore_background=True,
                                                 data_dir = dataset_path,
                                                 step_size = 5))

class Detectron2Wrapper:
    
    def __init__(self, cfg_path, model_path):
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        cfg.DATASETS.TEST = ('Mivos',)
        self.dataset_name = 'Mivos'
        self.dataset_path = '/home/kolz14w/haucode/hw2/mask_rcnn/Mivos_cus'
        self.predictor = DefaultPredictor(cfg)
        self.gen_meta()
    
    def gen_meta(self):
        data_reg(self.dataset_name, self.dataset_path)
        dataset = DatasetCatalog.get('Mivos')
        self.meta = MetadataCatalog.get('Mivos')
        num_categories = len(OBJ_CLASSES)
        self.meta.thing_classes = [OBJ_CLASSES[i] for i in range(num_categories)]
        
    def process(self, img_gbr):
        outputs = self.predictor(img_gbr)
        mask = outputs['instances'].get('pred_masks').to('cpu')
        classes = outputs['instances'].get('pred_classes').to('cpu')
        mask = mask.numpy()
        classes = classes.numpy()
        return mask, classes

cfg = Init_Save_Cfg()

# Set expected object classes
OBJ_CLASSES = {
    0: '00_hand',
    1: '01_cup',
    2: '02_spoon',
    3: '03_milk',
    4: '04_bowl',
    5: '05_Müsli',

}

INV_OBJ_CATEGORIES = {v: k for k,v in OBJ_CLASSES.items()}
print(INV_OBJ_CATEGORIES)

def Model_training(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.train()

det = Detectron2Wrapper('cfghw2_2.yaml', 'output/model_final.pth')

init_visualizer(det)



















