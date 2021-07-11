# Deep learning for predicting COVID-19 malignant progression      


### Introduction
This repository contains the code of the paper [Deep learning for predicting COVID-19 malignant progression](https://www.sciencedirect.com/science/article/pii/S1361841521001420?via%3Dihub). It develops an early-warning system with deep learning techniques to predict COVID-19 malignant progression. Our method leverages clinical data and CT scans of outpatients and achieves an AUC of 0.920 in the single-center study and an average AUC of 0.874 in the multicenter study.

<p align="center"><img width="75%" src="figures/Figure 1.png" /></p>
<p align="center"><img width="80%" src="figures/Figure 3.png" /></p>

### Prerequisites
* python 3.7
* pytorch 1.4
* numpy 
* torchnet
* imageio
* tqdm
* scikit-learn
* easydict
* pydicom

### How to Run

**Preparation**.
  1. Download pre-trained models:

     **Google Driver:** https://drive.google.com/drive/folders/1oKaoni5Jojs4kuo9R36_vuxTV2TIUWFg?usp=sharing    
     **Baiduyunpan:** https://pan.baidu.com/s/17JoFkZQ3BQTBvH57BiIL0g  **Password:** t4ay
       
  2. Modify the dataset path:

```Shell
config.data_root='/home/xxxx/works/COVID-19/PMP/data/
```
  3. Modify the pre-trained model path:

```Shell
config.pretrained_model_path=/home/xxxx/works/COVID-19/PMP/pretrained_model_dataset1/
```

  4. Copy datasets to your dataset path.
   
   
**Training**.  

For example, train our model on dataset 1

```Shell
python train_on_dataset1.py
```
It will save the models in ```./checkpionts/``` and results in ```./res/```.
   
   
**Testing**.  

For example, directly evaluate the model trained from dataset1 on dataset 2.

```Shell
python test_on_dataset2_baseline.py
```
It will save the results in ```./res/```.
   
   
**Domain adaptation**.  

For example, adapt the pre-trained model from dataset 1 on dataset 3 by a metric-learning method.

```Shell
python DA_on_dataset3.py
```
It will save the models in ```./checkpionts/``` and results in ```./res/```.

If you encounter any problems or have any inquiries, please contact cfang.meta@gmail.com.

**Citation**

If you use this code or use our pre-trained weights for your research, please cite our paper:
```
@article{Fang2021DeepLF,
  title={Deep learning for predicting COVID-19 malignant progression},
  author={Cong Fang and S. Bai and Qianlan Chen and Yu Zhou and Liming Xia and Lixin Qin and Shi Gong and Xudong Xie and Chunhua Zhou and Dandan Tu and Changzheng Zhang and Xiaowu Liu and Weiwei Chen and Xiang Bai and Philip H. S. Torr},
  journal={Medical Image Analysis},
  year={2021},
  volume={72},
  pages={102096 - 102096}
}
```
