import cv2
import torch
import urllib.request
import os
from tqdm.auto import tqdm
import numpy as np


import matplotlib.pyplot as plt

model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


IMGS_DIR = '../data/easyportrait/images/'
TARGET_DIR = '../data/easyportrait/depths/'
if not os.path.isdir(TARGET_DIR):
    os.mkdir(TARGET_DIR)

for subfolder in os.listdir(IMGS_DIR):
    depth_subfolder_path = os.path.join(TARGET_DIR, subfolder)
    if not os.path.isdir(depth_subfolder_path):
        os.mkdir(depth_subfolder_path)
    print(f'Processing {subfolder} ...')
    for filename in tqdm(os.listdir(os.path.join(IMGS_DIR, subfolder))):
        filepath = os.path.join(IMGS_DIR, subfolder, filename)
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        import pdb; pdb.set_trace()
        output_filename = os.path.join(depth_subfolder_path, filename.replace('.jpg', '.npz'))
        np.savez_compressed(output_filename, output)