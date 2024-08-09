import sys; sys.path.append('../../')
from models.SERSegFormer.model import SersegformerForSemanticSegmentation
from models.SERSegFormer.config import SersegformerConfig
from PIL import Image
import torch
import numpy as np
import cv2
from utils.pipeline import SemanticSegmentationPipeline

import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def build_pipeline(ckpt_path):
    config = SersegformerConfig.from_pretrained(ckpt_path)
    model = SersegformerForSemanticSegmentation.from_pretrained(ckpt_path)
    return SemanticSegmentationPipeline(model,depth=config.add_depth_channel) 

if __name__ == '__main__':
    # Create the pipeline
    ckpt_path = os.path.join(CURRENT_DIR, '../../checkpoints/serseg_abg_dbn_w_depth/')
    ckpt_path = os.path.join(CURRENT_DIR, '../../results/SersegformerForSemanticSegmentation_decoder/checkpoint-2000/')
    
    segmentation_pipeline = build_pipeline(ckpt_path)

    # Test the pipeline
    image = Image.open(os.path.join(CURRENT_DIR, "../../data/easyportrait/images/val/3cd3dd36-66f6-4a9e-af14-7112034dfcd1.jpg"))
    result = segmentation_pipeline(images=image)
    result = (result*255).astype('uint8').transpose((1,2,0)).squeeze(-1)
    Image.fromarray(result).save('pred.png')



    # With depth
    segmentation_pipeline = build_pipeline(ckpt_path)

    # Test the pipeline
    
    image = Image.open(os.path.join(CURRENT_DIR, "../../data/easyportrait/images/val/3cd3dd36-66f6-4a9e-af14-7112034dfcd1.jpg"))
    result = segmentation_pipeline(images=image)
    result = (result*255).astype('uint8').transpose((1,2,0)).squeeze(-1)
    Image.fromarray(result).save('pred_w_depth.png')

    
    
