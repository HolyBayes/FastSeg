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

    binary : bool
        convert the mask to binary (0 - background, 1 - person)

    depth : bool
        add depth map prediction if True  
    """
    def __init__(self, images_dir: str, annotations_dir: str,
                 transform: Optional[albu.Compose]=None, binary=True, depth=False):
        self.images = sorted(glob(os.path.join(images_dir, '*')))
        self.masks = sorted(glob(os.path.join(annotations_dir, '*')))
        self.transform = transform
        self.binary = binary
        
        self.depth_model = None
        if depth:
            depth_model_type = "MiDaS_small"

            self.depth_model = torch.hub.load("intel-isl/MiDaS", depth_model_type).to('cpu')
            self.depth_model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            if depth_model_type == "DPT_Large" or depth_model_type == "DPT_Hybrid":
                self.depth_transform = midas_transforms.dpt_transform
            else:
                self.depth_transform = midas_transforms.small_transform

        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.images[idx]
        image = cv2.cvtColor(cv2.imread(image_id), cv2.COLOR_BGR2RGB)
        add_depth = self.depth_model is not None
        if add_depth:
            with torch.no_grad():
                depth = self.depth_model(self.depth_transform(image))
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().detach().cpu().numpy()

        
        mask_id = self.masks[idx]
        mask = cv2.imread(mask_id, 0)
        if self.binary:
            mask = np.where(mask > 0, np.zeros_like(mask), np.ones_like(mask))
        
        if self.transform:
            if add_depth:
                sample = self.transform(image=image, mask=mask, depth=depth)
            else:
                sample = self.transform(image=image, mask=mask)
        image, mask, depth = sample['image'], sample['mask'], sample.get('depth')
        sample = {'image': image, 'mask': mask.long()}
        if add_depth:
            sample['depth'] = depth.unsqueeze(0)/1000
        return sample


class SegmentationDataCollator(DefaultDataCollator):
    def __call__(self, features):
        # import pdb; pdb.set_trace()
        pixel_values = torch.stack([f['image'] for f in features])
        labels = torch.stack([f['mask'] for f in features])
        batch_dict = {'image': pixel_values, 'mask': labels}
        if 'depth' in features[0]:
            batch_dict['depth'] = torch.stack([f['depth'] for f in features])
        return batch_dict
    

if __name__ == '__main__':
    from transforms import get_val_test_augmentation
    dataset = EasyPortraitDataset('./easyportrait/images/val/',
                                  './easyportrait/annotations/val/',
                                  get_val_test_augmentation(),
                                  binary=True,
                                  depth=True)
    sample = dataset[0]
    print(sample['image'].shape)
    print(sample['depth'].shape)