# NPMMR-Det
This is a PyTorch implementation of NPMMR-Det, YOLOv3, and YOLOv4.  

If you use it, please cite our paper and give this project a star. Thank you.

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

## Notice
The Lightweight versions (MobileNetv2, ShuffleNetv2, GhostNet...) will be available soon after our paper is published.  
The review speed is too slow!!!

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


### 最近很懒，以后慢慢写...有时间的话会出更详细的使用教程和每行代码的注释，先凑合着看吧

## 一些碎碎念（外国人就不要看了...虽然可能本来也没什么人看）
我初入遥感圈半年多，还在学习阶段，向各位大佬致敬，欢迎大家批评、交流，代码Bug请多包涵。  
最近看遥感目标检测论文看得实在想吐槽一下，论文里动不动就SOTA，SOTA您倒是开源代码开源模型啊，别怂。开源但复现不出结果的都算很良心了（某知名遥感目标检测方法，论文报道mAP70+，开源代码开源模型跑出来60+）。至于除了公开mAP再公开一下模型检测速度和复杂度的要求我都觉得太过分了。    
话说回来，遥感圈的巨佬们真是有钱，什么ResNet/ResNeXt-152/201的模型都能跑，刷榜刷不过你们我认（穷的瑟瑟发抖），求求你们慢一点，实在是跟不上，都是要发论文恰饭的，学术圈要以和为贵啊。  
NPMMR-Det这篇论文其实我自己很不满，谈不上多大创新，算一个增量性的工作，堆砌了一些花里胡哨的CV那边玩剩下的东西，欢迎多批评。入坑遥感目标检测半年多，看了一些论文，感觉大部分是在CV后面捡剩饭吃，拿过来换个数据就算创新，一些看起来明显就不讲武德的东西竟然能在遥感数据上涨点明显，这种炼丹技巧我也想学。至于各种train、val、test混合训练调参的我就不说啥了。  
由于各种这样那样的我不可抗力的原因，NPMMR-Det论文里没能把带方向旋转框OBB的结果放上挺遗憾的，我觉得这也算遥感目标检测里面为数不多有特色的东西了，实际我做了这个实验在NPMMR-Det的基础上，DOTA数据集的mAP=73.83，我会尽快在这个仓库里更新的修改代码和训练好的权重。  
![image](https://github.com/Shank2358/NPMMR-Det/blob/master/figs_readme/DOTA_OBB.png)  
由于某度网盘会员到期了，训练好的模型只传了Google Drive，下个月发工资我就续上。



