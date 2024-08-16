---
title:
layout: default
permalink: /research/GAN
published: true
---

# Table of contents

1. [What is Maximum Likelihood Estimation?](#0concept)

    1.1. [The Main Idea](#idea)

    1.2. [Diffusion Process](#process)

    1.3. [Model Architecture](#archi)

2. [What Is Diffusion Model?](#1concept)

    2.1. [The Main Idea](#1idea)

    2.2. [Diffusion Process](#1process)

    2.3. [Model Architecture](#1archi)

3. [Reference](#ref)


[DCGANs Notebook with document](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb#scrollTo=zB8Yhhy5Bc14)

## Recall Convolution Neural Networks: Upsampling, Downsampling, and Deconvolution [1]

> Tradition CNNs are used to compress and extract images' features.

#### 1. Upsampling, Downsampling and Dilation

\- Both of them is to expand/compress the input. Some techniques are widely used including padding (upsampling), stride (downsampling), dilations (downsampling).

\- In the *dilation*, the edge pieces of the kernel are pushed further away from the center piece.

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*SVkgHoFoiMZkjy54zM_SUw.gif)

#### 2. Transposed Convolution

\- is **upsampling** in nature. The layer will conduct operation on a modified input by calculating and adding 0's. [2]

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*KGrCz7aav02KoGuO6znO0w.gif)

\- **Applications**: are to reconstruct images (e.g., Generator in GANs, encoders,...)

\- Difference between ***Transposed Convolution*** vs ***Deconvolution***:

+ Deconvolution can reverse the Convolution output to get the exact same input, while Transposed Convolution does not give the same output as the input.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*gjDHvfY6XELWPZ50rqFs1A.png)

> **NOTE**: The effects of Downsampling and Upsampling will be reversed if they are applied to Transposed Convolution.


## Generative Adversarial Networks 

## Training

\- `Step 1`: **Discriminator**: tries to **maximize** $log(D(x)) + log(1-D(G(z)))$

\- `Step2`: **Generator**: tries to **minimize** $log(1-D(G(z)))$ or **maximize** $log((G(z)))$

\- The below notebook is an implementation of **DCGAN**, which uses convolution and transpose convolution layers in the Discriminator and Generator, respectively.

[Notebook with detailed documents](https://colab.research.google.com/drive/1LB8oNWhxnft9JauQIuGH1USeAClvsPuf#scrollTo=j6-ibf5iBc19)

## Reference

[1. Convolutions: Transposed and Deconvolution](https://medium.com/@marsxiang/convolutions-transposed-and-deconvolution-6430c358a5b6)

[2. What is Transposed Convolutional Layer?](https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11)