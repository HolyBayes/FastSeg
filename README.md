# Repository for fast person vs. background sem segmentation networks

# Repo structure

```
│
├── models/
│ └── SERSegFormer/
│ │ ├── config.py
│ │ ├── infer.py
│ │ └── model.py
│ ├── SERNet_Former/
│ │ ├── decoder.py
│ │ ├── encoder.py
│ │ ├── infer.py
│ │ └── model.py
│ └── SegFormer/
│ │ ├── config.py
│ │ ├── infer.py
│ │ └── model.py
│
├── modules/
│ ├── abg.py # Attention boosting Gate (from SERNet-Former)
│ ├── afn.py # Attention fusion Network (from SERNet-Former)
│ ├── dbn.py # Dilation based Network (from SERNet-Former)
│ ├── dam.py # Dual Attention Mechanism (from HSNet)
│ └── upsample.py # Nothing interesting here, just upsampling block of bilinear interpolation + Conv2d 
│
├── scripts/
│ ├── deploy.py # PyTorch -> ONNX -> TensorRT
│ ├── train_segformer.py # script for vanilla Segformer-B0 training
│ ├── train_sernet.py # script for vanilla SERNet training
│ ├── train_serseg.py # SERSeg
│ ├── train_serseg_decoder.py # Segformer-B0 with convolutional decoder
│ └── train_serseg_w_depth.py # Segformer-B0 with AbG, DBN, and Depth Map
│
├── utils/
│ ├── losses.py
│ ├── initialization.py
│ ├── metrics.py
│ ├── trainer.py
│ ├── visualization.py
│ └── pipeline.py # Basic Segmentation pipeline class
│
├── data/
│ ├── easyportrait/(.gitignored)
│ │ └── images/
│ │ └── annotations/
│ ├── transforms.py
│ └── dataset.py
│
├── checkpoints/ (.gitignored)
│ └── <checkpoint_files>
│
│
├── requirements.txt
└── README.md
```

# Inference

# Implemented models, modules, and features

## Models
- [x]  [SERNet-Former](https://arxiv.org/abs/2401.15741)
- [x]  [SegFormer](https://arxiv.org/abs/2105.15203)
- [x] SERSeg-Former (my model, which is a soup of latest sem seg and attention best practices on top of SegFormer-B0 backbone)  
- [x]  some additional modules like Dual Attention Mechanism from [HSNet](https://ieeexplore.ieee.org/document/10495017) as a bonus

## Modules

- [x] Attention-Boosting Gate (AbG, [https://arxiv.org/abs/2401.15741](https://arxiv.org/abs/2401.15741)) 
- [x] Attention fusion Network (AfN, [https://arxiv.org/abs/2401.15741](https://arxiv.org/abs/2401.15741))
- [x] Dual Attention Mechanism (DAM, [https://arxiv.org/abs/2401.15741](https://arxiv.org/abs/2401.15741))
- [x] Dilation Based Network (DbN, [https://ieeexplore.ieee.org/document/10495017
](https://ieeexplore.ieee.org/document/10495017)

## Features
- [x] CRF loss
- [x] DICE loss
- [x] SegFormer convolution decoder
- [x] Depth map
_
Note: all the features and modules are optional. It means you can manually enable or disable them in your personal configureation_

# Installation

```
pip install -r requirements.txt
```
_Note: Some extra dependencies may be included to the requirements.txt, which is a dummy snapshot of my working environment_

# Conducted experiments and TODOs

## SegFormer
- [x]  Replace Segformer's upsampling interpolation with convolutional Upsampling block

## SERNet

- [x]  Experiments with SERNet’s backbone ([SegFormer-B0](https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/mit.py#L246) w. [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/segformer-ps-512.pth))
- [x]  AbG now incorporates feature (channel) attention only, not any spatial (like ViT patches) attention ([SegFormer-B0](https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/mit.py#L246) w. [weights](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/easyportrait/experiments/models/segformer-ps-512.pth))
- [x]  Add learning-free [CRF](https://github.com/mishgon/crfseg/tree/master/crfseg) (Conditional Random Fields) layer to make the segmentation mask locally-consistent and smooth. The layer is learning-free and good even for the final mask post-processing. Easiest and chiepest way to gain some extra IoU, but may significantly slow down the inference. We can also add CRF to the training stage (say, to auxiliary heads and losses of the HSNet), if CRF significantly slows down the inference. *Note: takes 22ms on 512x512, so CRF loss only makes sense here. UPD: tested on SegFormer. Boost 0.1% (0.9795 →0.9796 IoU), does not worth it*
- [x]  Add channel compression and HSNet’s DAM to SERNet’s DbN *Note:*
- [x]  Add depth-maps as a separate channel (like in RGB-D) (https://pytorch.org/hub/intelisl_midas_v2/) (Adds 12ms to inference time, but may drastically increase the quality)
- [ ]  Add critic loss (CE and Dice loss values do not always reflect small segmentation artifacts, especially specific for the people vs. background segmentation problem)
- [x]  Add cross-attention to SERNet AfN’s AbGs with encoder feature maps (perform the skip-connection BEFORE AfN)
- [x]  Additional methods that are not applied during the SERNet-Former experiments, such as multiscale (MS) crop sizes of images as well as additional coarse datasets that most literature applies, can also improve the results of our network.Add cross-attention to SERNet AfN’s AbGs with encoder feature maps (perform the skip-connection BEFORE AfN)
- [x]  According to the authors discussion, the decoder part of SERNet-Former can also still be modified with AfNs.

# Numerical results
| Model | Inference time (ms) on 1080Ti (on raw PyTorch with no TensorRT and ONNX) | IoU x 100 | N_params (M) |
| --- | --- | --- | --- |
| SERNet-Former (EfficientNet) | 33 | 97.5 |  |
| SegFormer-B0 | 13 | 98.0 | 3.7 |
| SERSeg | ? (Needs way more training) | 4 |
|  |  |  |  |

# Conclusion
