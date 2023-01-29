# 3DCycleGAN
This modified version of CycleGAN[^1] was developed for enhancing the image quality of time-resolved, fast-acquisition microCT images of fibre-reinforced composites.
3D CycleGAN translates volumes between a low-quality, fast-acquisition domain (typically domain A) and a high-quality, slow-acquisition domain (typicalliy domain B). This enables an increased temporal resolution at constant image quality.
The results are discussed in [^2].

3DCycleGAN can denoise fast-acquisition tomograms to match the noise level of the respective slow-acquisition training set,
and it can apply 2x or 4x super-resolution.

## Quick-start
This section explains how to train and evaluate the algorithm with the datasets from this paper [^2].
### Preparation
1. Download this github repository.
2. Download the datasets from zenodo_link and copy them into the same folder.
3. We use [Anaconda](https://www.anaconda.com/) to manage our python packages. Hence [install Anaconda!](https://docs.anaconda.com/anaconda/install/)
4. Install the python environment `3DcycleGAN_env.yml` containing the relevant packages. (This will throw errors and can easily take 30 minutes.)
```
conda env create -f 3DcycleGAN_env.yml
```
If you have trouble with the environments, check [the documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) out.
### Training
1. Open the terminal and activate the anaconda environment.
```
conda activate 3DcycleGAN
```
2. Open `train.py` in a text/code editor and uncomment the relevant lines of code on the bottom of the file, e.g.:
```
## uncomment the following lines to train the 2x network for enhancement of the 1ms, 800nm dataset
    train(run_name = '2x_1ms',
        data_A = 'dataset/T700_GF_t05_pxs08.json',
        data_B = 'dataset/T700_GF_t05_pxs08.json',
        net_A = 'UNetSim3dSRx2',
        net_B = 'UNetSim3dSRd2',
        imsize_A = [128,128,128],
        super_resolution = 2)
```
3. Run `train.py`.
```
python3 train.py
```
4. Check the training progress.
`.../3DCycleGaN/results/*run_name*/train/` contains images of the cyclic translations:

![cyclic images](https://github.com/pvilla/3DCycleGaN/blob/main/imgs/trainCycle.png)

5. Stop the training when you are satisfied with the translation from `real_A` to `fake_B`.
```
Ctrl + C
```
### Evaluation
1. Open the terminal and activate the anaconda environment.
```
conda activate 3DcycleGAN
```

2. Open `evaluation.py` in a text/code editor and uncomment the relevant lines of code, e.g.:
```
## uncomment to evaluate the 2x network for enhancement of the 1ms, 800nm dataset
dfile = 
mfile = 
evaluate(datafile = dfile, modelfile = mfile, SR = 2, ev_name = '2x_1ms')
```
Make sure to change the location of `mfile` to suit your previously trained network.

The resulting enhancement is stored in `.../3DCycleGaN/eval/`.

## Understanding CycleGAN
CycleGAN is an unsupervised training strategy for generative neural networks. The algorithm learns how to translate images between two image domains.
The image below shows a flowchart of CycleGAN. 

![Flowchart of the cycleGAN algorithm.](https://github.com/pvilla/3DCycleGaN/blob/main/imgs/cycleGANflow6.png)

The flowchart shows the fast-acquisition domain on top and the slow-acquisition domain on the bottom.
The algorithm consists of four artificial neural networks (**generator fast->slow**, **discriminator slow**, **generator slow->fast**, **discriminator fast**).
A generator-discriminator set comprises one GAN[^3].
GANs are algorithms that can create photorealistic content. A generator network creates an image and tries to fool a discriminator network which differentiates between 'real' and generated images.
CycleGAN combines two GANs to enable cyclic image transformations.

Since we have two image domains, there are two possible starting points for image cycles.
The red arrows outline the **fast-slow-fast cycle**, where we  use **generator fast->slow** to translate *fast input* to *slow generated* and then use **generator slow->fast**, to convert *slow generated* back to the fast domain and create *fast cyclic*.
Conversely, the blue arrows outline the **slow-fast-slow cycle**, which performs the opposite domain translations.
Upon completion of each cycle, the input and cyclic image are compared to ensure that during the translations, no information was lost or added. We call this 'cycle consistency'.

We can follow the red **fast-slow-fast cycle** to understand how cycleGAN is optimized.
1. *fast input* is translated to *slow generated* with **generator fast->slow**
2. *slow generated* is evaluated with **discriminator slow**
3. **generator fast->slow** is updated to fool **discriminator slow** in the next training iteration
4. **discriminator slow** is uptdated with a generated and a 'real' input from the slow domain
5. *slow generated* is translated to *fast cyclic* with **generator slow->fast**
6. the distance between *fast input* and *fast cyclic* is computed with an L1 norm (cycle consistency)
7. **generator slow->fast** is updated to create an image that reduces the distance between *fast input* and *fast cyclic* given an input *slow generated*

The blue **slow-fast-slow cycle** is computed simulataneously in the same way.

**Like other GAN-based algorithms, CycleGAN is prone to introducing 'Hallucinations'. Hallucinations are invented artifacts that look like real image features but do not exist in the original input data. Hence, CycleGAN enhancements must be visually inspected and checked before they can be used for scientific data analysis.**

## How to ...
### ... find good hyperparameters?
CycleGAN is an algorithm that consists of four interdependent neural networks. These networks need to be balanced so that each network contributes to a successful training outcome.
The hyperparameters that we use to balance the networks are the weights for their loss functions and their learning rates.

### ... change the training datasets?
If you want to use 3D CycleGAN to enhance your own 3D datasets, you will have to prepare the data in the following way:

### ... modify the generator networks?
The generator networks in this version of cycleGAN are U-Nets[^4]. Compared to the original CycleGAN paper which uses U-Nets based on VGG11 or VGG16[^5], our networks are simplified to achieve an optimum between quality of the enhancement, memory usage and optimization time. A flow chart for our simplified U-Net without super resolution is depicted below.

![Flowchart of a U-Net for 1:1 translations.](https://github.com/pvilla/3DCycleGaN/blob/main/imgs/unetSIM2d.png)

The U-Net has only two pooling layers, which means that during an image translation, each pixel in the original image can only 'see' 8 pixels far. If your data contains different types of features with similar intensity transitions, you might want to consider a deeper U-Net.
In order to achieve super resolution we need an upscaling network for *generator fast->slow* and a downscaling network for *generator slow->fast*. This can be achieved by adding or removing encoder and decoder blocks as shown in the following schematic.

![Flowchart of downscaling and upscaling U-Nets for 2x super resolution.](https://github.com/pvilla/3DCycleGaN/blob/main/imgs/unetSR.png)

[^1]: https://junyanz.github.io/CycleGAN/ , https://arxiv.org/abs/1703.10593
[^2]: Not yet published. Link to our paper.
[^3]: cite goodfellow
[^4]: cite unet paper
[^5]: cite VGG paper
