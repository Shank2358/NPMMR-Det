# NPMMR-Det
This is a PyTorch implementation of NPMMR-Det

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



### 最近很懒，以后慢慢写...

## Weights
The pre-trained weights and trained models are available from [Google Drive](https://drive.google.com/drive/folders/1d9cT41TVg-Eae0CfMoPih8EgBMStZ4Jm?usp=sharing)

## License
This project is released under the [Apache 2.0 license](LICENSE).
