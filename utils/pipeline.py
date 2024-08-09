from PIL import Image
import torch
import numpy as np
import cv2



IMG_SIZE = (512,512)

class SemanticSegmentationPipeline(object):
    def __init__(self, model, size=IMG_SIZE, thresh=0, depth=False):
        self.model = model
        self.model.eval()
        self.size = size
        self.thresh = 0
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.depth = depth
        if self.depth:
            depth_model_type = "MiDaS_small"

            self.depth_model = torch.hub.load("intel-isl/MiDaS", depth_model_type).to('cpu')
            self.depth_model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            if depth_model_type == "DPT_Large" or depth_model_type == "DPT_Hybrid":
                self.depth_transform = midas_transforms.dpt_transform
            else:
                self.depth_transform = midas_transforms.small_transform
            

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
            input = torch.tensor(processed_images).permute(0, 3, 1, 2).float()
            if self.depth:
                depth = self.depth_model(self.depth_transform(image))
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=input.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().detach()
                depth = depth / 1000 # normalize the depth map from [0,1000] to [0,1] range
                outputs = self.model(input, depth=depth)
            else:
                outputs = self.model(input)
        outputs = outputs.detach().cpu().numpy().squeeze(axis=1)
        return np.where(outputs > 0, np.ones_like(outputs), np.zeros_like(outputs))

            


    def _resize(self, image, size):
        return cv2.resize(image, size)

    def _normalize(self, image, mean, std):
        image = image / 255.0
        image = (image - mean) / std
        return image