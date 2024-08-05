from torch.utils.data import Dataset
import os
from typing import Optional
import albumentations as albu
from glob import glob
import cv2
import numpy as np
from transformers import DefaultDataCollator
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(CURRENT_DIR, 'easyportrait/') # Put here your own path to dir with EasyPortrait dataset!

CLASSES = 9

IMGS_TRAIN_DIR = os.path.join(DATA_DIR, 'images/train')
ANNOTATIONS_TRAIN_DIR = os.path.join(DATA_DIR, 'annotations/train')

IMGS_VAL_DIR = os.path.join(DATA_DIR, 'images/val')
ANNOTATIONS_VAL_DIR = os.path.join(DATA_DIR, 'annotations/val')

IMGS_TEST_DIR = os.path.join(DATA_DIR, 'images/test')
ANNOTATIONS_TEST_DIR = os.path.join(DATA_DIR, 'annotations/test')




class EasyPortraitDataset(Dataset):
    """
    EasyPortrait Dataset class.
    
    Parameters
    ----------
    images_dir : str
        path to dir with images
        
    annotations_dir : 
        path to dir with annotated segmentation masks
        
    transform : albu.Compose
        composed list of data augmentation functions from albumentations
    """
    def __init__(self, images_dir: str, annotations_dir: str, transform: Optional[albu.Compose] = None, binary = True):
        self.images = sorted(glob(os.path.join(images_dir, '*')))
        self.masks = sorted(glob(os.path.join(annotations_dir, '*')))
        self.transform = transform
        self.binary = binary
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        image = cv2.cvtColor(cv2.imread(image_id), cv2.COLOR_BGR2RGB)
        
        mask_id = self.masks[idx]
        mask = cv2.imread(mask_id, 0)
        if self.binary:
            mask = np.where(mask > 0, np.zeros_like(mask), np.ones_like(mask))
        
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return {'image': image, 'mask': mask.long()}


class SegmentationDataCollator(DefaultDataCollator):
    def __call__(self, features):
        # import pdb; pdb.set_trace()
        pixel_values = torch.stack([f['image'] for f in features])
        labels = torch.stack([f['mask'] for f in features])
        return {'image': pixel_values, 'mask': labels}