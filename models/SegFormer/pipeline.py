import sys; sys.path.append('../../')
from models.SegFormer.model import SegformerForSemanticSegmentation
from models.SegFormer.config import SegformerConfig
from PIL import Image
import torch
import numpy as np
import cv2

import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE = (512,512)
CKPT_PATH = os.path.join(CURRENT_DIR, '../../results/SegformerForSemanticSegmentation/checkpoint-55000/')

class SegFormerPipeline(object):
    def __init__(self, model, size=IMG_SIZE, thresh=0):
        self.model = model
        self.model.eval()
        self.size = size
        self.thresh = 0
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, images):
        if not isinstance(images, list):
            images = [images]
        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                image = np.array(image)
            image = self._resize(image, size=self.size)
            image = self._normalize(image, mean=self.mean, std=self.std)
            processed_images.append(image)
        with torch.no_grad():
            outputs = self.model(torch.tensor(processed_images).permute(0, 3, 1, 2).float())
        outputs = outputs.detach().cpu().numpy().squeeze(axis=1)
        return np.where(outputs > 0, np.ones_like(outputs), np.zeros_like(outputs))

            


    def _resize(self, image, size):
        return cv2.resize(image, size)

    def _normalize(self, image, mean, std):
        image = image / 255.0
        image = (image - mean) / std
        return image
    
# Create the pipeline
segmentation_pipeline = SegFormerPipeline(SegformerForSemanticSegmentation.from_pretrained(CKPT_PATH))



if __name__ == '__main__':
    # Test the pipeline
    image = Image.open(os.path.join(CURRENT_DIR, "../../data/easyportrait/images/val/3cd3dd36-66f6-4a9e-af14-7112034dfcd1.jpg"))
    result = segmentation_pipeline(images=image)
    print(result)


#  The code needs to be refactored a bit to be compatible with Huggingface pipeline
#  from transformers import FeatureExtractionMixin, pipeline

# class SegFormerFeatureExtractor(FeatureExtractionMixin):
#     def __init__(self, size=IMG_SIZE):
#         self.size = size
#         self.mean = [0.485, 0.456, 0.406]
#         self.std = [0.229, 0.224, 0.225]

#     def __call__(self, images, *args, **kwargs):
#         return self.preprocess(images)

#     def preprocess(self, images):
#         processed_images = []
#         for image in images:
#             if isinstance(image, Image.Image):
#                 image = np.array(image)
#             image = self._resize(image, size=self.size)
#             image = self._normalize(image, mean=self.mean, std=self.std)
#             processed_images.append(image)
#         return {"pixel_values": torch.tensor(processed_images).permute(0, 3, 1, 2).float()}

#     def _resize(self, image, size):
#         return cv2.resize(image, size)

#     def _normalize(self, image, mean, std):
#         image = image / 255.0
#         image = (image - mean) / std
#         return image


# # Register the custom model and configuration
# SegformerForSemanticSegmentation.config_class = SegformerConfig
# SegformerForSemanticSegmentation.__name__ = "SegFormer"
# SegFormerFeatureExtractor.__name__ = "SegFormerFeatureExtractor"
