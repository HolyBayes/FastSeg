from transformers import FeatureExtractionMixin, pipeline
import numpy as np
from PIL import Image
import torch
import os, sys; sys.path.append('../../')
from models.SERNet_Former.model import SERNet_Former
from utils.pipeline import SemanticSegmentationPipeline
import cv2



CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    

def build_pipeline(ckpt_path):
    return SemanticSegmentationPipeline(SERNet_Former.from_pretrained(ckpt_path))

if __name__ == '__main__':
    CKPT_PATH = '../../checkpoints/sernet/'
    # Create the pipeline
    segmentation_pipeline = build_pipeline(CKPT_PATH)

    # Test the pipeline
    image = Image.open(os.path.join(CURRENT_DIR, "../../data/easyportrait/images/val/3cd3dd36-66f6-4a9e-af14-7112034dfcd1.jpg"))
    result = segmentation_pipeline(images=image)
    result = (result*255).astype('uint8').transpose((1,2,0)).squeeze(-1)
    Image.fromarray(result).save('pred.png')