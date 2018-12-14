# TensorFlow Playground with GAN 

## Image Generation with Generative Adversarial Networks

![ai-image-generator](assets/ai-image-generator.jpg)

## Getting Started

## Environment Setup

### Hardware

~~~
MacBook Pro 
macOS Mojave Version 10.14
OS Type 64-bit
Processor 2.5 GHz Intel Core i7
Memory
Graphics AMD Radeon R9 M370X 2048 MB/Intel Iris Pro 1536 MB
~~~

### Libraries and Software

Minimum requirements:

- pip3 
- python3.6
- numpy
- matplotlib
- tf-hub
- tensorflow

Check for dependecies:
~~~
pip3 --version
python3 --version   //minimum3.6
virtualenv --version
~~~

In case the dependencies have are not installed:
~~~
pip --upgrade
brew install python3.6
pip3 install numpy
sudo pip3 install -U virtualenv
~~~

Create the virtual environment 
~~~
mkdir ~/venv-tf
virtualenv ~/venv-tf
~~~

Activate virtual environment

~~~
source ~/venv-tf/bin/activate
~~~

To exit TensorFlow 

~~~
deactivate
~~~

Install TensorFlow pip package
~~~
pip3 install tensorflow-hub
pip3 install --upgrade tensorflow
python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl

~~~

TF-Hub Generative Image Model 

~~~~
pip3 install imageio
pip3 install scikit-image
pip3 install matplotlib.pyplot as plt

~~~~