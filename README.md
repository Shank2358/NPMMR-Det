# NPMMR-Det

## A Novel Nonlocal-aware Pyramid and Multiscale Multitask Refinement Detector for Object Detection in Remote Sensing Images  
(Refining Attention, Multiscale, and Multitask Misalignments of Object Detection in Remote Sensing Images)  

## This is a PyTorch implementation of [NPMMR-Det](https://ieeexplore.ieee.org/document/9364888), YOLOv3, and YOLOv4.  

## Citation
If you use it, please give this project a star and consider citing: 

@ARTICLE{9364888,  
  author={Z. {Huang} and W. {Li} and X. -G. {Xia} and X. {Wu} and Z. {Cai} and R. {Tao}},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},   
  title={A Novel Nonlocal-Aware Pyramid and Multiscale Multitask Refinement Detector for Object Detection in Remote Sensing Images},   
  year={2021},  
  volume={},  
  number={},  
  pages={1-20},  
  doi={10.1109/TGRS.2021.3059450}} 
  
Clone不Star，都是耍流氓~

## The lightweight and Oriented-BBoxes version [LO-Det](https://github.com/Shank2358/LO-Det) has been released (2021.3.31) 
## The NPMMRDet-Oriented_Bounding_Boxes version has also been released in [LO-Det](https://github.com/Shank2358/LO-Det) (2021.4.29)  

LO-Det: Lightweight Oriented Object Detection in Remote Sensing Images  

@ARTICLE{9390310,
  author={Z. {Huang} and W. {Li} and X. -G. {Xia} and H. {Wang} and F. {Jie} and R. {Tao}},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},   
  title={LO-Det: Lightweight Oriented Object Detection in Remote Sensing Images},   
  year={2021},  
  volume={},  
  number={},  
  pages={1-15},  
  doi={10.1109/TGRS.2021.3067470}}  



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

## 新开了一个tools文件夹,将会放一些数据格式转换的脚本和各种工具。  

## License
This project is released under the [Apache 2.0 license](LICENSE).

### 暑假會有大版本更新，提前預告一下。

### 最近很懒，以后慢慢写...有时间的话会出更详细的使用教程和每行代码的注释，先凑合着看吧

## 一些碎碎念，仅代表本人个人观点（外国人就不要看了...虽然可能本来也没什么人看）
初入遥感圈大半年，还在学习阶段，向各位大佬致敬，欢迎大家批评、交流，代码Bug请多包涵。  

NPMMR-Det这篇论文算一个增量性的工作，谈不上多大创新，是一个赶鸭子上架做的东西，堆砌了一些花里胡哨的CV那边玩剩下的东西，欢迎多批评。

由于各种这样那样的我不可抗力的原因，NPMMR-Det论文里没能把带方向旋转框OBB的结果放上挺遗憾的，我觉得这也算遥感目标检测里面为数不多有点特色的东西了，实际我做了这个实验在NPMMR-Det的基础上，DOTA数据集的mAP=73.83（544×544大小的输入，没有开多尺度测试，还可以刷到更高），我会尽快在这个仓库里更新的修改代码和训练好的权重。  
![image](https://github.com/Shank2358/NPMMR-Det/blob/master/figs_readme/DOTA_OBB.png)  
2020.3.1：过年期间调了调参，OBB的mAP能到76以上了，模型权重也传到网盘了，代码我还没封装好稍后更新。(4.29更新了)
![image](https://github.com/Shank2358/NPMMR-Det/blob/master/figs_readme/DOTA_OBB_new.png)  
 
 一些也许你们可以试试的涨点小技巧，我有时间了也会尝试这些然后把最新的结果在这里更新。  
（1）DOTA数据集给的都是大尺寸的整张图像，预处理其实挺重要的（比如剪裁、数据清洗筛选什么的），这一块一直没有统一标准很多论文也不会公开他们的训练和测试数据，不同的预处理数据对最后结果的影响挺大的。 裁剪train set的时候可以多裁减几种patch，比如600×600，1200×1200什么的，因为一些田径场、足球场目标比较大，裁剪600×600大小的patch可能都难以完整容下大部分这些目标，这会导致这些类别AP分低的可怜，（比如我的论文里就这样），但好像小目标会涨点。如果都用1200×1200这种大的，可能会爆显存（您卡多就当我没说，还见过某巨佬组卡多任性直接原图跑的，效果据报道非常不错，反正我这里没那么多卡测不起，没法验证是不是真的那么棒，有卡的盆友么可以试试，有结果了告诉我一声），裁剪这么大的然后模型里resize会导致小目标精度下降。好像也可以先把原图resize然后再剪裁，官方工具包里面好像有这样的函数。这个可以自己试验一下，有新发现了欢迎告诉我一声。  
（2）样本数量不均衡其实挺严重，可以在训练集中把一些比较少的目标的图片多复制几份让类别间尽量均衡（不一定是数量完全一样）。  
（3）NPMMR-Det的anchor是每层FPN有3个，三层共9个（论文里的anchor是kmeans算出来的，用的YOLO的代码），你们多搞几个anchor还可能涨点（这个改cfg文件就行），anchor的比例可以针对细长目标（桥、港口）多设置几种比例（1:2,1:4...）。   
（4）用更大的backbone，在./model/backbone里面我也放了ResNet什么的各种大模型，预训练权重我会尽快放出来，卡多就试试看呗，会有惊喜。  
（5）把多尺度测试打开，多尺度测试的尺度多选几种（NPMMR-Det论文里用的是416-608，步长96，也就是416×416，512×512，608×608，不妨把96改小，也就是多几种尺度，还会涨点）。  
（6）one-stage的老问题，正负样本标签分配那里可以再改改，用”软“阈值做一下分配，可以参考一下guided-anchor和ATSS。另外最大检测数量可以再改大如果有显存的话。另外anchor分配里面有一个祖传bug，我注释出来了，不过对实验结果没影响。  
（7）DIOR数据集小目标（小车车）是真的小，可以训练的时候选择大一点的输入尺寸，原图是800×800的，NPMMR-Det论文和代码用的reshape后的最大尺寸（开了多尺度训练）是640×640，你们可以在cfg里面改大一点试试看。  
（8）dataset.py里面藏了很多其他在我们的论文里没用到的数据增强tricks(NPMMR-Det论文只开了尺度变化、平移、仿射、然后HSV),你们可以打开其他的然后去试试顺便调个参(旋转变换那里有个越界bug我正在修，暂时下线)，有好的调参结果烦请在这里分享一下。  
（9）最新一些即插即用的模块还有attention也实现了一些，参考了SimpleCVReproduction这个项目，用这些彩蛋试试看吧，有好结果了欢迎分享，发论文了跪求引用一下我的论文，谢谢啊。  
（10）NMS現在代碼裏用的是python-numpy版本的，速度比較慢但是兼容各個系統。CUDA加速版本的在Lib文件夾底下的倉庫裏，linux的make一下然後就可以用了，import一下然後把原來代碼裏的NMS函數換成那個就行。windows的目前加速版還有bug，我最近沒空修改，你們可以去github搜一下有沒有其他倉庫的實現，替換一下就行。  
其他一些没研究明白的东西，欢迎讨论：  
（1）DOTA的train set和val set的数据分布是不一样的，我简单统计了一下。然后val set和test set应该也是不一样，而且差的不少据我推测，因为我跑出来的现象是val set上面的分数是低于上传平台测出来的test set的分数的，然后我试过val上面不是取最优结果的权重去做测试，系统给的分数反而更高了。最后我没用val去调参，直接固定最大epoch，其他参数和对比论文设置的一样（别问参数怎么选的，我做梦梦见的），所有实验都测试最后五个epoch的结果取平均，懒得调参了，爱咋咋滴，反正卷不过。你们可以调参看看，应该是还会涨点的。  
（2）DOTA数据集的直升机（HC）我的模型一直检测的不好，分挺低的，目前还没研究明白为什么，有的论文竟然70多分这一类，谁知道为什么烦请告诉我。   
（3）代码中多卡并行的那里还有一点点bug我正在修...  
（4）pytorch 1.6及以上模型的权重那里有可能会保存成.zip形式，再用低版本打开需要转换一下，不过你们应该是只用一个版本就行了不需要像我去测试不同版本，万一遇到了就看一下官方手册，由于大家用的版本可能不一样。  
（5）Windows系统下编译DCN要记得装一个Microsoft Visual C++ Build Tools(微软官网可以下载)，有遇到BUG就在issues里面给我留言。Linux系统GCC就行，记得更新一下。    
（6）torch转ONNX和TensorRT的代码快写哭我了（好多要从头写），这个短期内更新不了，科研论文党没有影响直接先拿这个程序去用吧没啥影响，工业应用党我暂时不推荐你们用这个仓库。有没有大佬可以帮帮我，我会在这个项目里把你的名字加上。  
（7）一些优化和工程化的代码由于项目原因没有办法开源，有点遗憾。

由于某度网盘会员到期了，训练好的模型只传了Google Drive，下个月发工资我就续上。 

挖了个新坑，这个仓库不定期更新，有急事的话论文里的邮箱可以在工作时间联系到我，不急的可以在issues里面留言。



