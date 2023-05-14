# AI_Seminar_Assignment-1_TransPose
This is a repository about the Assignment 1 of subject AI Seminar 2023-1. The aim of this assignment is to run a published deep learning project successfully and perform a minor modification to the original code.

## Introduction
[TransPose](https://github.com/yangsenius/TransPose) is a Human Pose Estimation(HPE) project established by Yang, Sen, et al. in International Conference on Computer Vision 2021. This project implements Transformer Encoder in existed HPE deep learning model. They aimed to further help the understanding of spatial dependencies the model captures to localize the human's keypoints. The attention layers built in Transformer encoder enable the capturing of long-range relationships efficiently. The attention map further reveal what dependencies the predicted keypoints rely on. 

[Paper](https://arxiv.org/abs/2012.14214)

### Modification
I made some minor modifications based-on the default TransPose project. First, I applied Automatic Mixed Precision(AMP) to speed up the learning of the model while overcoming the limitation of computing resources. AMP enables the model to be trained with half precision and consumes less memory during the training while maintaining consistent accuracy. The AMP is implemented using the built-in torch.amp package. For more details about the AMP please refer to https://developer.nvidia.com/automatic-mixed-precision and https://pytorch.org/docs/stable/amp.html.

The second modification is the Ground truth heatmap. The default ground truth heatmaps are constructed by covering a 2D Gaussian kernel onto the keypoints with standard deviation of 2 and corresponding keypoint's coordinate as its centre. I modified the value of standard deviation to 1.5 which the activated area of the heatmap will become relative smaller. In the CNN-based baseline HPE model such as [SimpleBaseline](https://github.com/microsoft/human-pose-estimation.pytorch) and [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation), model trained using ground truth heatmap with standard deviation of 1.5 achieved a slightly performance boost. However, from my experiment, I achieved a totally different result.😮

### Results on COCO val2017 

|     Model      | Input size | Sigma | AP    | Ap .5 | AP .75 | AP (M) | AP (L) |  AR   | AR .5 | AR .75 | AR (M) | AR (L) |
| :------------: | :--------: | ----- | ----- | ----- | :----: | :----: | :----: | :---: | :---: | :----: | :----: | :----: |
| TransPose-R-Enc3 |  256x192   | 2.0 | 0.723 | 0.915 | 0.795  | 0.692  | 0.769  | 0.753 | 0.925 | 0.815  | 0.718  | 0.804  |
| TransPose-R-Enc3 |  256x192   | 1.5 | 0.720 | 0.914 | 0.792  | 0.688  | 0.769  | 0.751 | 0.921 | 0.815  | 0.715  | 0.805  |



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

I installed all packages from anaconda using _conda install_ as possible since I am using conda environment. For exmaple,
> conda install -c anaconda pandas

However, there is an **exception** when installing opencv-python. The opencv-python package installed using _conda install -c conda-forge opencv_ may causing the import failure. Therefore, here I suggest to install opencv through pip:
>pip install opencv-python

If your device is running on  Ubuntu Server and **no** open-GL related driver is installed, use _pip install opencv-python-headless_ instead. 

### Dataset Preparation
For download of COCO dataset, please refer to [here](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9).

TransPose follow the the steps of [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch#data-preparation) to prepare the COCO train/val/test dataset and the annotations. The detected person results are downloaded from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing). Please download or link them to ${POSE_ROOT}/data/coco/, and make them look like this:

```txt
$preferedlocation/coco/
|-- annotations
|   |-- person_keypoints_train2017.json
|   `-- person_keypoints_val2017.json
|-- person_detection_results
|   |-- COCO_val2017_detections_AP_H_56_person.json
|   `-- COCO_test-dev2017_detections_AP_H_609_person.json
`-- images
	|-- train2017
	|   |-- 000000000009.jpg
	|   |-- ... 
	`-- val2017
		|-- 000000000139.jpg
		|-- ... 
```

## Run
Change the location of dataset and pretrained models in /experiment/coco/transpose_{prefered model}/*.yaml

### Run in terminal
#### Train on COCO train2017 dataset
```bash
python tools/train.py --cfg experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml
```
### Run in PyCharm
#### Train on COCO train2017 dataset
Modify path of config file in "train.py" and run the file.
