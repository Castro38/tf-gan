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

## Environment Setup



## Implement the Image Generation Model


![animation](assets/animation.gif)

## How human are the generated Images?

## Summary


## Futher Reading 

- [The Mostly Complete Chart of Neural Networks, explained](https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464)
- [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Hub](https://www.tensorflow.org/hub/)