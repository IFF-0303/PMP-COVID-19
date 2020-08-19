# Predicting COVID-19 malignant progression with AI techniques
By [Cong Fang], [Song Bai].

### Introduction
This repository contains the code of the paper [Predicting COVID-19 malignant progression with AI techniques](https://arxiv.org/abs/xxxx.xxxxx). It develops an early-warning system with deep learning techniques to predict COVID-19 malignant progression. Our method leverages clinical data and CT scans of outpatients and achieves an AUC of 0.920 in the single-center study and an average AUC of 0.874 in the multicenter study.

### Prerequisites
* python 3.7
* pytorch 1.4.0
* numpy 
* torchnet
* imageio
* tqdm
* scikit-learn
* easydict
* pydicom

### How to Run

**Train on dataset 1**. For example, train a MLP + 3D ResNet model trained with cross-entropy loss

```Shell
python train_on_dataset1.py
```
It will save the adversarial images and features.

**Test on dataset 2/3 directly**.

```Shell
python test_on_dataset2/3_baseline.py
```
Shell in one trial. We support three attacking methods, including FGSM, I-FGSM and MI-FGSM.

**Domain adaptation on dataset 3**.

```Shell
python DA_on_dataset3.py
```
Shell in one trial. We support three attacking methods, including FGSM, I-FGSM and MI-FGSM.

### Citation and Contact

If you find the code useful, please cite the following paper

    @article{bai2020adversarial,
      title={Predicting COVID-19 malignant progression with AI techniques},
      author={Bai, Xiang and Fang, Cong and Zhou, Yu and Bai, Song and Torr, Philip HS},
      journal={arXiv preprint arXiv:xxxx.xxxxx},
      year={2020}
    }

If you encounter any problems or have any inquiries, please contact cfang.meta@gmail.com.







