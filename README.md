# Predicting COVID-19 malignant progression with AI techniques
By [Cong Fang], [Song Bai].

### Introduction
This repository contains the code of the paper [Predicting COVID-19 malignant progression with AI techniques](https://arxiv.org/abs/xxxx.xxxxx). It develops an early-warning system with deep learning techniques to predict COVID-19 malignant progression. Our method leverages clinical data and CT scans of outpatients and achieves an AUC of 0.920 in the single-center study and an average AUC of 0.874 in the multicenter study.

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
  1. Download datasets and pre-trained models:

     **Google Driver:** https://drive.google.com/    
     **Baiduyunpan:** https://pan.baidu.com    
     **Password:** 11ee
  
  2. Modify the dataset path:

```Shell
config.data_root='/home/xxxx/works/COVID-19/PMP/data/
```
  3. Modify the pre-trained model path:

```Shell
config.pretrained_model_path=/home/xxxx/works/COVID-19/PMP/pretrained_model_dataset1/
```

  4. Cpoy datasets to your dataset path.
   
   
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
