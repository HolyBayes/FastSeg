# Repository for fast person vs. background sem segmentation networks

# Structure

# Inference

# Implemented models, modules, and features

## Models
- [x]  [SERNet-Former](https://arxiv.org/abs/2401.15741)
- [x]  [SegFormer](https://arxiv.org/abs/2105.15203)
- [x]  some additional modules like Dual Attention Mechanism from [HSNet](https://ieeexplore.ieee.org/document/10495017)

## Modules

## Features
- [x] CRF loss
- [x] DICE loss
- [x] SegFormer convolution decoder
- [x] Depth map
_
Note: all the features and modules are optional. It means you can manually enable or disable them in your personal configureation_

## Modules

# Installation

```
pip install -r requirements.txt
```
_Note: Some extra dependencies may be included to the requirements.txt, which is a dummy snapshot of my working environment_

# Experiments


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

# Results

# Deployment
