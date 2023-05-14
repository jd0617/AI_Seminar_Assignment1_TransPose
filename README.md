# AI_Seminar_Assignment-1_TransPose
This is a repository about the Assignment 1 of subject AI Seminar 2023-1. The aim of this assignment is to run a published deep learning project successfully and perform a minor modification to the original code.

## Introduction
[TransPose](https://github.com/yangsenius/TransPose) is a Human Pose Estimation(HPE) project established by Yang, Sen, et al. in International Conference on Computer Vision 2021. This project implements Transformer Encoder in existed HPE deep learning model. They aimed to further help the understanding of spatial dependencies the model captures to localize the human's keypoints. The attention layers built in Transformer encoder enable the capturing of long-range relationships efficiently. The attention map further reveal what dependencies the predicted keypoints rely on. 

### Modification
I made some minor modifications based-on the default TransPose project. First, I applied Automatic Mixed Precision(AMP) to speed up the learning of the model while overcoming the limitation of computing resources. AMP enables the model to be trained with half precision and consumes less memory during the training while maintaining consistent accuracy. The AMP is implemented using the built-in torch.amp package. For more details about the AMP please refer to https://developer.nvidia.com/automatic-mixed-precision and https://pytorch.org/docs/stable/amp.html.

The second modification is the Ground truth heatmap. The default ground truth heatmaps are constructed by covering a 2D Gaussian kernel onto the keypoints with standard deviation of 2 and corresponding keypoint's coordinate as its centre. I modified the value of standard deviation to 1.5 which the activated area of the heatmap will become relative smaller. In the CNN-based baseline HPE model such as [SimpleBaseline](https://github.com/microsoft/human-pose-estimation.pytorch) and [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation), model trained using ground truth heatmap with standard deviation of 1.5 achieved a slightly performance boost. 

## Environment, Framework and Installation
### Environment and Framework
- OS: Ubuntu 20.04
- GPU: RTX 3090 x 1
- Environment: conda environment
- Framework (version): PyTorch (1.13.1)
- Dataset: COCO dataset
- Required packages: same as original [TransPose](https://github.com/yangsenius/TransPose/blob/main/requirements.txt)
### Installation
For installation of packages, please refer to [original repository](https://github.com/yangsenius/TransPose).

I installed all packages from anaconda as possible since I am using conda environment. For exmaple,
> conda install -c anaconda pandas

However, there will be an exception when installing opencv-python. The opencv-python package installed using _conda install -c conda-forge opencv_ may causing the import failure. Therefore, here I suggest to install opencv through pip:
>pip install opencv-python

If your device is running on  Ubuntu Server and **no** open-GL related driver is installed, use _pip install opencv-python-headless_ instead. 

### Dataset Preparation
For download of COCO dataset, please refer to [here](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9).

TransPose follow the the steps of [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch#data-preparation) to prepare the COCO train/val/test dataset and the annotations.
