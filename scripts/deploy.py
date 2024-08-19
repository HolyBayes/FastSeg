import numpy as np
np.bool = np.bool_
np.int = np.int_
import torch
import sys; sys.path.append('../')
from models.SERSegFormer.model import SersegformerForSemanticSegmentation
import coremltools as ct
from utils.pipeline import SegmentationDeploymentDecorator




IMG_SIZE = 512
TORCH_CKPT_DIR = '../checkpoints/serseg_w_decoder/'
ONNX_PATH = "../checkpoints/serseg_w_decoder.onnx"
COREML_PATH = '../checkpoints/serseg_w_decoder.mlpackage'
TEST_IMG_PATH = '..//data/easyportrait/images/val/0a3f6908-e244-4636-8c7b-1b27d7cdadce.jpg'

def main():
    # Load the pre-trained PyTorch model
    torch_model = SegmentationDeploymentDecorator(
        SersegformerForSemanticSegmentation.from_pretrained(TORCH_CKPT_DIR)
    )
    # torch_model_input = torch.randn((1,3,IMG_SIZE,IMG_SIZE)) # Unnormalized RGB image [0,255], channel first

    import cv2
    img = cv2.imread(TEST_IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    torch_model_input = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)

    
    # PyTorch -> ONNX
    onnx_model = torch.onnx.dynamo_export(torch_model, torch_model_input)
    onnx_model.save(ONNX_PATH)

    traced_model = torch.jit.trace(torch_model, torch_model_input)

    out = traced_model(torch_model_input)
    
    
    # Convert prediction back to readable format
    mask_pred = out[0][0].detach().cpu().numpy()
    mask_pred = np.where(mask_pred > 0, np.ones_like(mask_pred), np.zeros_like(mask_pred))
    mask_pred = 1-np.repeat(np.expand_dims(mask_pred, -1), 3, -1)
    mask_pred = (255*mask_pred).astype('uint8')

    # from PIL import Image
    # Image.fromarray(mask_pred).save('./example_pred.png')

    # ONNX -> CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=torch_model_input.shape)]
    )
    coreml_model.save(COREML_PATH)
    

if __name__ == "__main__":
    main()
