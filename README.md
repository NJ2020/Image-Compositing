# Image Compositing

#### *Note: This project was done by me during the summer internship last summer, at Samsung Research Institute. It addresses an intriguing problem which I think is quite relevant from an industrial research point of view, especially for companies that are working on AI-powered smartphones.*

___

## Aim
>**To generate aesthetically pleasing, photorealistic composite by merging two selfies.**

*You might be thinking, what does that even mean?*

*Let me try to make things simpler and explain it to you in layman language. Suppose we have two selfies, so what we want to do is, extract any one subject (usually pointed out by the user through a click) from the first selfie and paste it **automatically** at the best possible location in the second selfie (considering factors like aspect and zoom ratio, etc. ) without making the final composite look unnatural.*

Now, if you do a proper thought experiment, this seems to be a very complicated task since finding the best location in the second selfie among the many possible ones, while also keeping in mind the *spatial relationship between the subject and background (second selfie itself)* is not a trivial thing to do. The algorithm should be very robust to ***scale and rotational invariance***. Assuming we somehow found the solution to the above problem, still, we have another major difficulty in our way. The *lighting condition of the two selfies can be completely different (one can be in broad light, another one in backlight or one can be in bright daylight and another one in low dim nightlight).* So, we also need to match the **lighting conditions** of the background (second selfie) and foreground (subject) in a way that the overall composite looks natural.

____

## Motivation

*One can imagine many potential use cases of it, but I would like to state few, which I can think off the top of my head. We can do all the things mentioned below, in **Adobe Photoshop**, but that would require a lot of **manual labor, technical skills and a high inference time**.*

- People can create and share their memories just like a collage, for their **personal use.**
- **Anime production** can use it to create CGI scenes by skipping the intermediate step of the 3D rendering of an object.   

___

## Problem Breakdown

The exciting part about the project was that neither has this **Image to Image translation** problem been explored much by Deep Learning nor are there any publicly available datasets for Image Compositing. So, as I started pondering on how to approach this complex task at hand, I decided to divide the whole project into two parts. 

- Running an **instance segmentation** over the first selfie to extract the relevant subject from it and then, to automatically place it at the most suitable position (using some algorithm) in the second selfie, where it would not look out of place.
- Using **Conditional GANs** to *match the lighting conditions and other statistical features* of the foreground (subject) and background (second selfie) to give it a *photo-realistic touch.*

### Dataset Creation

> Due to time constraints, I took a slight detour, spent a considerable amount of time creating my custom dataset using **Microsoft COCO**, which was later used as an input for the second half. Creating a meaningful dataset while keeping the final aim of the project in mind was the most challenging task since it was the only way to get the best possible proxy for the first part. Following is a brief overview of what I did: 

- Picked any two random images from the whole dataset (original COCO) in which the category people covered a significant portion of the image (should be uncrowded too). 
- Masked a subject from the first image and edited its appearance using Image processing techniques like *random perturbation in brightness/contrast/saturation (method_1), and color transfer (method_2)* from the target object (with same semantic) of the second image.

*This way the output will be a composite in which the statistical features of one of the subjects will be completely different compared to that of its background and, can act as a perfect substitute for an object extracted from the first selfie and pasted into another at the best possible location.*

Now, let's do a much more detailed analysis of the second method used for creating the custom dataset. The method used was the **Style transfer** proposed by the authors of [this](https://arxiv.org/pdf/1511.03748.pdf) paper to produce compelling, and artifact-free stylized output. It can be broken down into the following three steps: 

#### Pre-processing:

*To effectively stylize images with global transforms,*
- Compress the dynamic ranges of the two images using a γ (= 2.2) mapping and convert the images into the LAB colorspace (because it decorrelates the different channels well). 
- Stretch the luminance (L channel) to cover the full dynamic range after clipping both the minimum and the maximum 0.5 percent pixels of luminance levels. 
- Apply the different transfer functions to the luminance and chrominance components as described below.

#### Chrominance:
Color transfer method used here maps the statistics of the chrominance channels of the two images. The chrominance distribution of an image is modeled using a ***multivariate Gaussian***, and a transfer function is designed that creates the output image O by mapping the Gaussian statistics N (µS, ΣS) of the style exemplar S to the Gaussian statistics N (µI , ΣI ) of the input image I as: 
``` 
C_output(x) = T (C_input(x) − µI ) + µS s.t. T * Σ_original_img * transpose(T)  = Σ_style_img 
```
, where T is a linear transformation that maps chrominance between the images and, C_input(x) is the chrominance at pixel x of the input image. The solution is unstable for low input covariance values, leading to color artifacts when the input has low color variation. To avoid this, regularize the solution by clipping diagonal elements of Σ_original_img as 
``` 
Σ_original_img = max(Σ_original_img , λrI)
```

##### Luminance:
- They matched contrast and tone using histogram matching between the luminance channels of the input and style exemplar images. Direct histogram matching typically results in arbitrary transfer functions and may produce artifacts due to non-smooth mapping or excessive stretching/compressing of the luminance values. Instead, they designed a new parametric model of luminance mapping that allows for strong expressiveness and regularization simultaneously. The transfer function is defined as: 

![1](https://user-images.githubusercontent.com/41862477/50593661-31e2cb80-0ebf-11e9-8128-b7d123ad2cba.jpg)

- They extracted a luminance feature, L, that represents the luminance histogram with uniformly sampled percentiles of the luminance cumulative distribution function (32 samples). They estimated the tone-mapping parameters by minimizing the cost function: 

![1](https://user-images.githubusercontent.com/41862477/50593706-7b331b00-0ebf-11e9-9df2-c71d43f0828f.jpg)

Results:

![1](https://user-images.githubusercontent.com/41862477/50594272-e847b000-0ec1-11e9-99ac-f49add58878a.jpg)
![3](https://user-images.githubusercontent.com/41862477/50594273-e8e04680-0ec1-11e9-807a-9c51a53b2bc2.jpg)
![img_0](https://user-images.githubusercontent.com/41862477/50594274-e8e04680-0ec1-11e9-97f6-376921271abf.jpg)

*Few images from the Dataset:*

![1](https://user-images.githubusercontent.com/41862477/49573700-aa0fad00-f964-11e8-8466-7bd4d7780fa5.JPG)
![2](https://user-images.githubusercontent.com/41862477/49573701-aa0fad00-f964-11e8-84a9-2ca1f3d8da40.JPG)
![3](https://user-images.githubusercontent.com/41862477/49573703-aaa84380-f964-11e8-8225-2d0c02ef89ac.JPG)

COCO dataset has originally around 40,000 images in which category **people** covered a significant portion of the image. I tried my best to ensure that, the composites are neither arbitrary nor unrealistic in color and tone by carrying out the editing of the appearance of the subject very carefully. Sadly, there were still a few images that one would ideally want to discard from his/her dataset while training the Deep Learning model (**Garbage In Garbage Out** is one of the most common phrases in Machine Learning). But, It will be difficult for one to go through every image and filter out the bad composite, as it is a tedious and repulsive task to do.

> *So, to overcome those issues, I trained a **discriminative** Deep CNN which learned the perception of visual realism in terms of **color, lighting and texture compatibility**, from a large amount of **unlabelled data**. It was able to make a proper distinction between the natural images and automatically generated image composites.* By using this model, I was able to discard very unnatural looking composites from the raw custom dataset. *Now, finally, we have a collection of composites in which the edited object still looks plausible, but does not match the background context.*

### Training ConditionalGAN:

Given the dataset and final aim in mind, I reduced the remaining task to **Image to Image translation** problem (Motivated by this astounding paper [Pix2pix](https://arxiv.org/pdf/1611.07004.pdf)). In analogy to automatic language translation, automatic image-to-image translation refers to the task of translating one possible representation of an image into another, given sufficient training data. I deployed **Conditional Generative Adversarial networks** not only to learn the mapping from the input image to the output image but also to learn a loss function to train this mapping. This generic approach alleviated the issues that would traditionally require very different loss formulations (Image compositing has already been thoroughly investigated by Image Processing community, and there are numerous proposals to get some decent results by following some specific, hand designed approach. However, they aren't scalable at all for obvious reasons).

___

## Pix2pix

> ***If you are unfamiliar with GANs, I would request you to look into some of the GAN tutorials and then proceed forward.***

From now on, I will only be giving you the overview of **Pix2pix** paper and the changes (mostly in the architecture and loss function) that I made to make the model work properly; remaining details are nearly the same. Like other GANs, Conditional GANs also have one discriminator (or critic depending on the loss function you are using) and one generator, and it tries to learn a conditional generative model which makes it suitable for image-to-image translation tasks, where we condition on an input image and generate a corresponding output image. 

> If mathematically expressed, CGANs learn a mapping from observed image X and random noise vector z, to y, *G : {x, z} → y*. The generator G is trained to produce outputs that cannot be distinguished from **real** images by an adversarially trained discriminator, D, which in turn is itself optimized to do as well as possible at identifying the generator’s **fakes**.

### Loss Function

The objective of a conditional GAN can be expressed as:

> **```Lc GAN (G, D) = Ex,y (log D(x, y)) + Ex,z (log(1 − D(x, G(x, z)))```** where G tries to minimize this objective against an adversarial D that tries to maximize it, i.e. **```G∗ = arg min(G)max(D) Lc GAN (G, D)```**. It is beneficial to mix the GAN objective with a more traditional loss, such as L1 distance to make sure that, the ground truth and the output are close to each other in L1 sense **```L(G) = Ex,y,z ( ||y − G(x, z)|| )```**.

Without z, the net could still learn a mapping from x to y, but would produce deterministic outputs, and therefore fail to match any distribution other than a **delta function**. Instead, the authors of Pix2pix provided noise only in the form of **dropout**, applied on several layers of the generator at **both training and test time**.

The Min-Max objective mentioned above was used in the original paper when GAN was first proposed by **Ian Goodfellow** in 2014, but unfortunately, it doesn't perform well due to vanishing gradients problems. Since then, there has been a lot of development, and many researchers have proposed different kinds of loss formulation (LS-GAN, WGAN, WGAN-GP) to overcome these issues. 

I played around with two such loss functions as mentioned below: 
- **Least-Square GAN loss**, and
- **Wasserstein GAN loss** (both with and without Gradient penalty)

> Minimizing Least Square GAN loss in the optimization procedure yielded me better results, in a far lower number of training iterations compared to Wasserstein GAN loss. WGAN was very slow to train (because of the multiple updates needed to be done for the critic for each update of the generator) and there were also some weird artifacts in the output composite.  

However, there was a recent paper by **Google** which addressed this issue and stated that "None of the loss functions is optimal in every scenario", it's always task dependent (maybe WGAN can perform better than LS-GAN in one task, while the other way around has also equally likely chance to happen in some different scenario). So, before switching to a different loss function, you should instead focus on extensive hyper-parameter optimization. Training GANs is very tricky, and it will never work in the first attempt, you might even want to look into the capacity of your architecture. There was this, one recent theoretical paper on GANs by [Sanjeev Arora](https://arxiv.org/abs/1706.08224) in which he mentioned that the generator's minimum capacity should be as large as the capacity of the discriminator squared. 

### Network Architecture

#### Generator: 

In Image-to-image translation problems, we map a high resolution input grid to a high resolution output grid. Both are renderings of the same underlying structure with the only difference in the surface appearance. The authors designed the generator architecture around these considerations. They used an encoder-decoder network in which the input is passed through a series of layers that progressively downsample, until a bottleneck layer, at which point the process is reversed. 

> To preserve the low-level details, skip connections were used. Specifically, skip connections are added between each layer i and layer n − i, where n is the total number of layers. Each skip connection simply concatenates all channels at layer i with those at layer n − i.

```Architecture: 
Encoder:  E(64, 1) - E(64, 1) - E(64, 2) - E(128, 2) - E(256, 2) - E(512, 2) - E(512, 2) - E(512, 2) - E(512, 2)
Decoder:  D(512, 2) - D(512, 2) - D(512, 2) - D(256, 2) - D(128, 2) - D(64, 2) - D(64, 2) - D(64, 1) - D(3, 1)
```

#### Discriminator:

The GAN discriminator models high-frequency structure term, relying on an L1 term to force low-frequency correctness. In order to model high-frequencies, it is sufficient to restrict the attention to the structure in local image patches. Therefore, discriminator architecture was termed PatchGAN – that only penalizes structure at the scale of patches. This discriminator tries to classify if each N × N patch in an image is real or fake. We run this discriminator convolutionally across the image, and average all responses to provide the ultimate output of D.

> Instead of using vanilla convolutional layers to increase the receptive field (or decrease the size of the image), I tried using **WideResNet-Block** in my experiments with the instance normalization sliced between the conv layers. This worked better than from what suggested in the original paper; maybe we actually need a larger number of trainable parameters for the discriminator in this case. The receptive field was also, that of the full image as opposed to 70 * 70 (used originally).

The original Patch-gan was giving some weird artifacts due to the fact that, it was optimized for some different task (Authors pointed out that receptive field of 70 * 70 was giving best results, compared to any smaller or larger receptive field). Patch GANs discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter.

```Architecture: WRB64 - WRB128 - WRB256 - WRB512```

___

## Results

![1](https://user-images.githubusercontent.com/41862477/49601764-bfa8c500-f9ac-11e8-83ec-355d77cb4d5c.JPG)
![2](https://user-images.githubusercontent.com/41862477/49601766-bfa8c500-f9ac-11e8-92a2-19625692fcd8.JPG)
![3](https://user-images.githubusercontent.com/41862477/49601767-c0415b80-f9ac-11e8-8546-392744c8a696.JPG)

![out_1](https://user-images.githubusercontent.com/41862477/49601769-c0415b80-f9ac-11e8-8931-14423a5c9016.JPG)
![out_2](https://user-images.githubusercontent.com/41862477/49601770-c0415b80-f9ac-11e8-8721-55d6ed024eb1.JPG)
![out_3](https://user-images.githubusercontent.com/41862477/49601771-c0d9f200-f9ac-11e8-9f16-40cb608648b4.JPG)

___

***Thanks for going through this post! Any feedbacks are duly appreciated.***
