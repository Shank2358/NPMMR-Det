# NPMMR-Det
This is a PyTorch implementation of NPMMR-Det

## Clone不Star， 都是耍流氓！

## Environments
Linux & Windows

1. For RTX20/Titan RTX/V100 GPUs  
cudatoolkit==10.0.130  
numpy==1.17.3  
opencv-python==3.4.2  
pytorch==1.2.0  
torchvision==0.4.0  
pycocotools==2.0 (In the ./lib folder)  
dcnv2==0.1 (In the ./lib folder)  
...  
The installation of other libraries can be carried out according to the prompts of pip/conda  
  
2. For RTX30 GPUs （I only tested it on RTX3090 GPUs currently ）  
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
You need to write a script to convert them into the train.txt file required by this repository and put them in the ./data folder.  
For the specific format of the train.txt file, see the example in the /data folder.

## Usage Example
1. train  
python train.py  
2. test  
python test.py  

## Parameter Settings
Modify ./cfg/cfg_npmmr.py, please refer to the comments in this file for details

## Weights
The pre-trained weights and trained models are available from [Google Drive](https://drive.google.com/drive/folders/1d9cT41TVg-Eae0CfMoPih8EgBMStZ4Jm?usp=sharing)  
Put them in. /weight folder

### 最近很懒，以后慢慢写...有时间的话会出更详细的使用教程和每行代码的注释，先凑合着看吧

## Notice
The Lightweight versions (MobileNetv2, ShuffleNetv2, GhostNet...) will be available soon after our paper is published.  
The review speed is too slow!!!

If you have any questions, please ask in issues.  
If you find bugs, please let me know and I will debug them in time. Thank you.
I will do my best to help you run this program successfully and get results close to those reported in the paper.  

## Something New
In addition to YOLOv3, YOLOv4 has also been initially implemented in this repository.  
Some of the plug-and-play modules (many many Attentions, DGC, DynamicConv, PSPModule, SematicEmbbedBlock...) proposed in the latest papers are also collected in the ./model/plugandplay, you can use them and evaluate their performance freely. If it works well, please share your results here. Thank you.  

## References
https://github.com/Peterisfar/YOLOV3  
https://github.com/argusswift/YOLOv4-pytorch  
https://github.com/ultralytics/yolov5  
https://github.com/pprp/SimpleCVReproduction  

## License
This project is released under the [Apache 2.0 license](LICENSE).
