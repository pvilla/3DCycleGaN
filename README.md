# 3DCycleGAN
This modified version of CycleGAN[^1] was developed for enhancing the image quality of time-resolved, fast-acquisition microCT images of fibre-reinforced composites.
3D CycleGAN translates volumes between a low-quality, fast-acquisition domain (typically domain A) and a high-quality, slow-acquisition domain (typicalliy domain B). This enables an increased temporal resolution at constant image quality.
The results are discussec in [^2].

3DCycleGAN can denoise fast-acquisition tomograms to match the noise level of the respective slow-acquisition training set,
and it can apply 2x or 4x super-resolution.

## Quick-start
This section explains how to train and evaluate the algorithm with the datasets shown in paper [^2].


## Understanding CycleGAN
CycleGAN is an unsupervised training strategy for generative neural networks. The algorithm learns how to translate images between two image domains.
The image below shows a flowchart of CycleGAN. 

![Flowchart of the cycleGAN algorithm.](https://github.com/pvilla/3DCycleGaN/blob/main/imgs/cycleGANflow6.png)

The algorithm consists of four artificial neural networks (generator fast-slow, discriminator slow, generator slow-fast, discriminator fast). A set of 

## How to ...
### ... change the training datasets?
### ... modify the generator networks?
### ... find good hyperparameters?



[^1]: https://junyanz.github.io/CycleGAN/ , https://arxiv.org/abs/1703.10593
[^2]: Not yet published. Link to our paper.
