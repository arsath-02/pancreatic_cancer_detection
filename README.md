# Let's create the README.md file with the provided content in code format.

readme_content = """
# Image Classification with Various Deep Learning Models

This project demonstrates the training and evaluation of various deep learning models for image classification using TensorFlow and Keras. The models include MLP, CNN, DenseNet121, ResNet50, InceptionV3, MobileNet, and VGG19. Each model is trained with and without data augmentation.

## Table of Contents

- [Introduction](#introduction)
- [Models](#models)
  - [MLP (Multi-Layer Perceptron)](#mlp-multi-layer-perceptron)
  - [CNN (Convolutional Neural Network)](#cnn-convolutional-neural-network)
  - [DenseNet121](#densenet121)
  - [ResNet50](#resnet50)
  - [InceptionV3](#inceptionv3)
  - [MobileNet](#mobilenet)
  - [VGG19](#vgg19)
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project applies different deep learning architectures to image classification tasks. Each model is trained on a custom dataset with and without data augmentation to compare their performance.

## Models

### MLP (Multi-Layer Perceptron)

The MLP is a basic neural network used for image classification without convolutional layers. It processes flattened images directly. This model is trained without data augmentation.

### CNN (Convolutional Neural Network)

CNNs are specialized neural networks designed to process data with grid-like topology, such as images. They consist of convolutional layers, pooling layers, and fully connected layers. This model is trained with and without data augmentation.

### DenseNet121

DenseNet121 is a convolutional neural network that connects each layer to every other layer in a feed-forward fashion. This helps to improve the flow of gradients and feature reuse. This model is trained with and without data augmentation.

### ResNet50

ResNet50 is a deep residual network that uses shortcut connections to jump over some layers, mitigating the vanishing gradient problem in very deep networks. This model is trained with and without data augmentation.

### InceptionV3

InceptionV3 is a deep convolutional neural network architecture that uses inception modules to allow the model to learn rich feature representations. This model is trained with and without data augmentation.

### MobileNet

MobileNet is a lightweight model optimized for mobile and embedded vision applications. It uses depthwise separable convolutions to reduce the number of parameters and computational cost. This model is trained with and without data augmentation.

### VGG19

VGG19 is a deep convolutional network with 19 layers. It uses very small convolution filters and has a simple and uniform architecture. This model is trained with and without data augmentation.

## Prerequisites

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Data Preparation

The dataset consists of CT images of pancreatic cancer, categorized into healthy and unhealthy classes. The dataset should be organized into training and testing directories, with subdirectories for each class. The images should be resized to the appropriate input size for each model (e.g., 224x224).

## Installation

Clone the repository and install the required packages:

```sh
git clone https://github.com/arsath-02/pancreatic_cancer_detection.git
cd pancreatic_cancer_detection

