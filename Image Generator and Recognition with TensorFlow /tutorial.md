---
title: Image Generation and Recognition with TF
author: sandbox-team
tutorial-id: xxx
experience: Beginner
persona: Data Scientist
source: Hortonworks
use case: Example (Data Discovery)
technology: TensorFlow
release: hdp-3.0.1
environment: Sandbox
product: HDP
series: TBD
---

# Generating and Labeling Images with TF

## Introduction

This tutorial covers basic concepts of Deep Convolutional Generative Adversarial Networks or DCGANs and the role it plays in the generation of images.

You will learn to generate images and detect how "human" those generated images are using Googles TensorFlow while leveraging the Hortonworks Data Platform(HDP) sandbox.

![frame-15-labeled](assets/frame-15-labeled.jpg)

## Prerequisites

- Downloaded and deployed the [Hortonworks Data Platform (HDP)](https://hortonworks.com/downloads/#sandbox) Sandbox
- [Object Detection in 5 Minutes](http://hortonworks.com//tutorial/object-detection-in-5-minutes)
- [Model Retraining for Object Recognition](http://hortonworks.com//tutorial/model-retraining-for-object-recognition)

## Outline

- [Concepts](#concepts)
- [Environment Setup](#environment-setup)
- [Implement the Image Generation Model](#implement-the-image-generation-model)
- [How human are the generated Images?](#how-human-are-the-generated-images?)
- [Summary](#summary)
- [Further Reading](#further-reading)
- [Appendix A: Troubleshoot](#appendix-a-troubleshoot)
- [Appendix B: Extra Features](#appendix-b-extra-features)

## Concepts

### GANs

Amongst the plethora of Machine learning neural networks there has been one network that has not ceased to generate fuzz. This very famous network is the Generative Neural Network or GAN's. GANs was created in 2014 by Goodfellow, refer to the **Further Reading** section to learn more about GAN's from the source.

The GAN architecture is composed of two neural networks who are engaged in an epic zero-sum battle. Both networks start with an original set of data, in this case we will use images. On one side there is Neural Network-1 who's sole purpose is to train with the original data set and generate images from noise that can fool  Neural Network-2 to think that the images are from the original data set. Meanwhile Neural Network-2 trains with the original data set to detect any fakes that Neural Network-1 sends its way and give the probability that the image given is real or fake.

Both Networks are initially trained separately then put up against each other, during this process each network learns to either create better fakes or discriminate the real from the fake. From a perspective of statistics, was the image set generated X = {x<sub>1</sub>,...,x<sub>n</sub>} and X<sup>'</sup>={x<sup>'</sup><sub>1</sub>,...,x<sup>'</sup><sub>n</sub>} drawn from the same distribution? If not,the Generative Network will train to create data that resembles the distribution.

### DCGANs

The Deep Convolutional Generative Adversarial Networks or DCGAN is another type of generative network. Unlike GAN, DCGANs is composed of convolutional layers  without max pooling or fully connected layers

#### Gradient Descent

#### Latent Space

#### The Data Set

### Environment Setup

The following libraries need to be installed:

- python3
- numpy
- matplotlib
- tf-hub
- tensorflow

Check for the following dependencies:
```
pip3 --version
python3 --version   //minimum3.6
virtualenv --version
```

If you have not done so install the libraries below:

```
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv
```

```
virtualenv --system-site-packages -p python3 ./venv # From home direcotry
source ./venv/bin/activate
```

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"

pip3 install numpy
pip3 install matplotlib
pip3 install tensorflow
pip3 install tensorflow-hub
pip3 install --upgrade tensorflow
python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
```

## Implement the Image Generation Model

### Install Dependencies for the Image Generator

TF-Hub Generative Image Model

```
pip3 install imageio
pip3 install scikit-image
pip3 install matplotlib.pyplot as plt
pip3 install ipython
pip3 install numpy
/Applications/Python\ 3.6/Install\ Certificates.command
```

In your terminal go to the **ven** directory you created.

```
cd ~/venv
```

Download the python code to generate celebrity images.

>Note: Learn more about Google's Colab [Image Generation Python Script](https://github.com/tensorflow/hub/blob/master/examples/colab/tf_hub_generative_image_module.ipynb)

~~~
wget github.com/gen-image.py
~~~

Run the image generator code:

~~~
python3 gen-image.py
~~~

### Results
You should see the following results:

~~~
~/venv
$python3 gen-image.py
2018-12-19 17:55:19.249869: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
<IPython.core.display.Image object>
0.68876916
5.607328
5.049135
2.7256749
0.7315421
1.4969518
0.8285036
1.1469646
1.4737974
0.78105783
0.87805945
1.2609681
0.8296515
0.5128517
0.7658002
0.2812683
1.0127488
1.3108158
0.95765066
0.1757649
0.3891008
0.22717363
0.20078099
0.379921
0.14679064
0.8292588
0.9534563
0.5001564
0.641171
0.94268334
0.6888621
0.25469738
0.40378252
0.17268972
0.18588635
0.356967
0.16601633
0.6979192
0.79671913
0.37014747
(venv)
~~~

Also check your **venv** directory for the animation gif generated with celebrity images.

![animation](assets/animation.gif)

## How human are the generated Images?

DCGAN networks are very powerful as seen on the results on the previous section. 


![frame-00-labeled](assets/frame-00-labeled.jpg)
![frame-06-labeled](assets/frame-06-labeled.jpg)
![frame-15-labeled](assets/frame-15-labeled.jpg)
![frame-20-labeled](assets/frame-20-labeled.jpg)

## Summary
Congratulations, you have succesfully gn



## Futher Reading 

- [The Mostly Complete Chart of Neural Networks, explained](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464)
- [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Hub](https://www.tensorflow.org/hub/)