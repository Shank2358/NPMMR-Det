# NPMMR-Det

## A Novel Nonlocal-aware Pyramid and Multiscale Multitask Refinement Detector for Object Detection in Remote Sensing Images (Refining Attention, Multiscale, and Multitask Misalignments of Object Detection in Remote Sensing Images)  

This is a PyTorch implementation of NPMMR-Det, YOLOv3, and YOLOv4.  

If you use it, please cite our paper and give this project a star. Thank you.

## Cloning this project without starring is a hooliganism
## Clone不Star，都是耍流氓！

## Environments
Linux (Ubuntu 18.04, GCC>=5.4) & Windows (Win10, VS2019)   
CUDA 11.1, Cudnn 8.0.4

1. For RTX20/Titan RTX/V100 GPUs （I have tested it on RTX2080Ti, Titan RTX, and Tesla V100 (16GB)）  
cudatoolkit==10.0.130  
numpy==1.17.3  
opencv-python==3.4.2  
pytorch==1.2.0  
torchvision==0.4.0  
pycocotools==2.0 (In the ./lib folder)  
dcnv2==0.1 (In the ./lib folder)  
...  
The installation of other libraries can be carried out according to the prompts of pip/conda  
  
2. For RTX30 GPUs （I have tested it on RTX3080 and RTX3090 GPUs）  
cudatoolkit==11.0.221  
numpy==1.17.5  
opencv-python==4.4.0.46  
pytorch==1.7.0  
torchvision==0.8.1  
pycocotools==2.0 (In the ./lib folder)  
dcnv2==0.1 (In the ./lib folder)  
...

## Installation
1. git clone this repository    
2. Install the libraries in the ./lib folder  
(1) DCNv2  
cd ./NPMMR-Det/lib/DCNv2/  
sh make.sh  
(2) pycocotools  
cd ./NPMMR-Det/lib/cocoapi/PythonAPI/  
sh make.sh  

## Datasets
1. [DOTA dataset](https://captain-whu.github.io/DOTA/dataset.html) and its [devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
2. [DIOR dataset](https://pan.baidu.com/share/init?surl=w8iq2WvgXORb3ZEGtmRGOw), password: 554e  
(1) VOC Format  
You need to write a script to convert them into the train.txt file required by this repository and put them in the ./data folder.  
For the specific format of the train.txt file, see the example in the /data folder.
(2) MSCOCO Format  
put the .json file in the ./data folder.

## Usage Example
1. train  
python train.py  
2. test  
python test.py  

## Parameter Settings
Modify ./cfg/cfg_npmmr.py, please refer to the comments in this file for details

## Weights
The pre-trained weights and trained models are available from [Google Drive](https://drive.google.com/drive/folders/1d9cT41TVg-Eae0CfMoPih8EgBMStZ4Jm?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1FQD7nUmXKTIljmiMHeHslg) (password: gxzx)  
Put them in. /weight folder

## Notice
The Lightweight versions (MobileNetv2, ShuffleNetv2, GhostNet...) will be available soon after our paper is published.  
The review speed is too slow!!! By the way, some reviewers have been struggling with some simple and well-known mathematical common sense (such as Taylor's expansion), which really makes me speechless. Maybe they prefer the circuit-connected CNN model.

If you have any questions, please ask in issues.  
If you find bugs, please let me know and I will debug them in time. Thank you.  
I will do my best to help you run this program successfully and get results close to those reported in the paper.  

## Something New
In addition to YOLOv3, YOLOv4 has also been initially implemented in this repository.  
Some of the plug-and-play modules (many many Attentions, DGC, DynamicConv, PSPModule, SematicEmbbedBlock...) proposed in the latest papers are also collected in the ./model/plugandplay, you can use them and evaluate their performance freely. If it works well, please share your results here. Thank you.

## To Do
(1) YOLOv4: Mosaic Data Augmentation    
(2) Better attention visualization: CAM & Grad-CAM  
(3) Guided Anchor  
(4) Model Pruning  
(5) ONNX & TensorRT  
(6) Transformer Head  
(7) More Backbones & pre-trained weights (SE-ResNet, ResNeSt, RegNet...)  
(8) NAS  
...  

## References
https://github.com/Peterisfar/YOLOV3  
https://github.com/argusswift/YOLOv4-pytorch  
https://github.com/ultralytics/yolov5  
https://github.com/pprp/SimpleCVReproduction  

## License
This project is released under the [Apache 2.0 license](LICENSE).

