## Compensation Atmospheric Scattering Model and Two-Branch Network for Single Image Dehazing

Xudong Wang, Xi’ai Chen *, Weihong Ren, Zhi Han, Huijie Fan, Yandong Tang, Lianqing Liu <br />
 <br />
The State Key Laboratory of Robotics, Shenyang Institute of Automation, Chinese Academy of Sciences, and also University of Chinese Academy of Sciences (UCAS). (e-mail: wangxudong@sia.cn).
## Our work 

We deduce a reformulated atmospheric scattering model for a hazy image and propose a novel lightweight two-branch dehazing network. In the model, we use a Transformation Map to represent the dehazing transformation and use a Compensation Map to represent variable illumination compensation. Based on this model, we design a Two-Branch Network (TBN) to jointly estimate the Transformation Map and Compensation Map. Our TBN is designed with a shared Feature Extraction Module and two Adaptive WeightModules. The Feature Extraction Module is used to extract shared features from hazy images. The two Adaptive Weight Modules generate two groups of adaptive weighted features for the Transformation Map and Compensation Map, respectively. This design allows for a targeted conversion of features to the Transformation Map and Compensation Map. To further improve the dehazing performance in the real-world, we propose a semi-supervised learning strategy for TBN. Specifically, by performing supervised pre-training based on synthetic image pairs, we propose a Self-Enhancement method to generate pseudo-labels, and then further train our TBN with the pseudo-labels in a semi-supervised way. Extensive experiments demonstrate that the model-based TBN outperforms the state-of-the-art methods on
various real-world datasets.

<p float="left">
  &emsp;&emsp; <img src="./f.png" width="900" />
</p>

## Dependencies
* Python 3.8
* PyTorch 1.8.1 + cu111
* torchvision 0.9.1 + cu111
* numpy
* opencv-python
* skimage
* hiddenlayer
* matplotlib
* PIL
* math
* os
## Architecture
model.py: The definition of the model class.

utils.py: Some tools for network training and testing.

data.py: Preparation tools for the training dataset.

test.py: Quick dehazing test for hazy images.

testall.py: Dehazing test for all hazy images dataset.

train.py: Training the dehazing model by supervised learning.

SemiStrain.py: Training the dehazing model by Semi-supervised learning in specific dataset.


## Test
1. Please put the images to be tested into the ``test_images`` folder. We have prepared the images of the experimental results in the paper.
2. Please run the ``test.py``, then you will get the following results:
<p float="left">
  &emsp;&emsp; <img src="./f2.png" width="900" />
</p>

## Test all
If you want to test the results on a labeled dataset such as [O-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/) , you can go through the following procedure:
1. Please put the dataset to be tested into the ``test0`` folder. You need put the hazy images into the ``test0/hazy`` folder, and put the clear images into the ``test0/gt`` folder. We have prepared the dataset of the experimental results in the paper.
2. Please run the ``testall.py``, then you will get the dehazing results SSIM, PSNR, and Inference time.

## Train
You can perform supervised learning of the network by following this step.
1. Please put the dataset into the ``train_data`` folder. You can get the [RESIDE](https://sites.google.com/view/reside-dehaze-datasets) for training.
2. Please run the ``train.py``, then you will get the dehazing model in ``saved_models`` folder.

## Semi-supervised Train
You can perform semi-supervised learning of the network by following this step.
1. Please make sure you have got the supervised learning trained model.
2. Please put the specific dataset into the ``Sdata/gt_hazy`` folder, which does not require any image with labels. 
3. Please run the ``SemiStrain.py``, then you will get the Semi-supervised learning dehazing model in ``saved_models`` folder.

## Citation
Thank you for your interest and welcome to cite our paper:
```bibtex
@ARTICLE{10504912,
  author={Wang, Xudong and Chen, Xi'ai and Ren, Weihong and Han, Zhi and Fan, Huijie and Tang, Yandong and Liu, Lianqing},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence}, 
  title={Compensation Atmospheric Scattering Model and Two-Branch Network for Single Image Dehazing}, 
  year={2024},
  volume={},
  number={},
  pages={1-17},
  doi={10.1109/TETCI.2024.3386838}}
```
