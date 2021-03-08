# PEN-Net
Reproducing "Learning Pyramid-Context Encoder Network for High-Quality Image Inpainting" of CVPR 2019 by tensorflow  

## Environmental Requirements
* tensorflow 2.0  

## Using
train  
> python run.py /root/image_root_path/ /root/mask_root_path  

test  
> python test.py /root/image_root_path/ /root/mask_root_path  

All parameters are set in config.py.

## Details  
The main innovation of this paper lies in the feature filling from the characteristics of the upper layer to the lower layer. 
