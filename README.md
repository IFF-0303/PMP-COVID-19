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

**Train**. For example, train a MLP + 3D ResNet model trained with cross-entropy loss

```Shell
python Gen_Adv.py \
 --loss_type=soft \
 --name=resnet_50 \
 --save_img \
 --save_fea
```
It will save the adversarial images and features.

**Test**.

```Shell
python evaluate_adv.py \
 --loss_type=soft \
 --name=resnet_50
```
Shell in one trial. We support three attacking methods, including FGSM, I-FGSM and MI-FGSM.

```bash
sh adv.sh
```
### Visualizations

#### Visualizations of Adversarial Examples

<p align="left"><img src="Images/1.png" width="176"> <img src="Images/2.png" width="176"> <img src="Images/3.png" width="176"> <img src="Images/4.png" width="176"></p>

#### Visualizations of Ranking List
<p align="left">
<img src="Images/untarget_illustration-crop-1.png" alt="Non-targeted Attack" width="720px">
</p>

The ranking list of non-targeted attack.

<p align="left">
<img src="Images/target_illustration_cropped-1.png" alt="Targeted Attack" width="720px">
</p>

The ranking list of targeted attack.

### Citation and Contact

If you find the code useful, please cite the following paper

    @article{bai2020adversarial,
      title={Predicting COVID-19 malignant progression with AI techniques},
      author={Bai, Xiang and Fang, Cong and Zhou, Yu and Bai, Song and Torr, Philip HS},
      journal={arXiv preprint arXiv:xxxx.xxxxx},
      year={2020}
    }

If you encounter any problems or have any inquiries, please contact songbai.site@gmail.com or songbai@robots.ox.ac.uk







