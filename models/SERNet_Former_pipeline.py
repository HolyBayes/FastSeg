from transformers import FeatureExtractionMixin, pipeline
import numpy as np
from PIL import Image
import torch
from models.SERNet_Former import SERNet_Former, SERNetConfig

class SERNetFeatureExtractor(FeatureExtractionMixin):
    def __init__(self, size=(1024,768)):
        self.size = size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, images):
        return self.preprocess(images)

    def preprocess(self, images):
        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                image = np.array(image)
            image = self._resize(image, size=self.size)
            image = self._normalize(image, mean=self.mean, std=self.std)
            processed_images.append(image)
        return {"pixel_values": torch.tensor(processed_images).permute(0, 3, 1, 2)}

    def _resize(self, image, size):
        return np.array(Image.fromarray(image).resize((size, size)))

    def _normalize(self, image, mean, std):
        image = image / 255.0
        image = (image - mean) / std
        return image


# Register the custom model and configuration
SERNet_Former.config_class = SERNetConfig
SERNet_Former.__name__ = "SERNet"
SERNetFeatureExtractor.__name__ = "SERNetFeatureExtractor"


CKPT_PATH = 'checkpoints/latest.pth'
# Create the pipeline
segmentation_pipeline = pipeline(
    task="image-segmentation",
    model=SERNet_Former.from_pretrained(CKPT_PATH),
    feature_extractor=SERNetFeatureExtractor(size=(1024,768)),
)

# Test the pipeline
image = Image.open("path/to/your/image.jpg")
result = segmentation_pipeline(images=image)
print(result)