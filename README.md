# AI_Seminar_Assignment-1_TransPose
This is a repository about the Assignment 1 of subject AI Seminar 2023-1. The aim of this assignment is to run a published deep learning project successfully and perform a minor modification to the original code.

## Introduction
TransPose(https://github.com/yangsenius/TransPose) is a Human Pose Estimation(HPE) project established by Yang, Sen, et al. in International Conference on Computer Vision 2021. This project implements Transformer Encoder in existed HPE deep learning model. They aimed to further help the understanding of spatial dependencies the model captures to localize the human's keypoints. The attention layers built in Transformer encoder enable the capturing of long-range relationships efficiently. The attention map further reveal what dependencies the predicted keypoints rely on. 

### Modification
I made some minor modifications based-on the default TransPose project. First, I applied Automatic Mixed Precision(AMP) to speed up the learning of the model while overcoming the limitation of computing resources. AMP enables the model to be trained with half precision and consumes less memory during the training while maintaining consistent accuracy. The AMP is implemented using the built-in torch.amp package. For more details about the AMP please refer to https://developer.nvidia.com/automatic-mixed-precision and https://pytorch.org/docs/stable/amp.html.
