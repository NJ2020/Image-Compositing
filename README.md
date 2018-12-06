# Image Compositing

#### *Note: This work was done by me during the internship at Samsung Research Institute, last summer. It addresses an intriguing problem which I think is quite relevant from an industrial research point of view, especially for companies that are working on AI-powered smartphones.*

## Aim
>**In short, the aim was to generate aesthetically pleasing, photorealistic composite by merging two (or more) selfies.**

*You might be thinking, what does that even mean?*

Let me try to make things simpler and explain it to you in layman language with the help of an example. Suppose we have two selfies, so what we want to do is, extract any one individual (usually pointed out by the user through a click) from the first selfie and paste it automatically at the best possible location in the second selfie, without making the composite look unnatural. 

>*I will be using the word subject to refer to the individual extracted from the first selfie from now on, to keep the discussion manageable.* 

Now, if you start thinking about it, this seems to be a very complicated task since finding the best location among many possible ones in the second selfie, while keeping in mind the spatial relationship between the subject and background (second selfie) is not a trivial thing to do. The algorithm should be very robust to scale and rotational invariance. 

Assuming we somehow managed to do the first part, still, we have another major challenge to solve - The lighting condition of the two selfies can be completely different (one can be in broad light, another one in backlight or one can be in bright daylight and another one in low dim nightlight). So, we also need to match the lighting conditions of the background (second selfie) and foreground (subject) in a way that the overall composite looks natural. It should be clear by looking at the composite that both the individuals were present while clicking the selfie.

>***So, this was the final aim and hope that I was able to explain it clearly.*** 

## Motivation

*One can think of many potential use cases of it, but I would like to state few which I can think off, the top of my head. We all can do these things mentioned below in Photoshop but that would require a lot of manual labor, and considerable inference time.*

- People can create and share their memories just like a collage for their **personal use.**

- **Anime industry** can use it to create CGI scenes by skipping the intermediate step of the 3D rendering of an object.   

## Problem Breakdown
The exciting part about the project was that neither has this **Image to Image translation** problem been explored much by Deep Learning
nor are there any publicly available datasets for Image Compositing. So, as I started pondering on how to approach this complex task at hand, I decided to divide the whole project into two parts. 

- Running an **instance segmentation** over the first selfie to extract the relevant subject from it and then, to automatically place it at the most suitable position in the second selfie, where it would not look out of place.
- Using **Conditional GANs** to match the lighting conditions and other statistical features of the foreground (subject) and background (second selfie) to give it a photo-realistic touch.

> Due to time constraints, I took a slight detour, spent a considerable amount of time creating my custom dataset using **COCO** data, which I can use as an input to the latter half. Creating a meaningful dataset while keeping the final aim of the project in mind was the most challenging task since it was the only way to get the best possible proxy for the first part. Following is a brief overview of what I did: 

### Dataset Creation

- Picked any two random images from the whole dataset (original COCO) in which the category people covered a significant portion of the image. 
- Masked a subject from the first image and edited its appearance using Image processing techniques like *random perturbation in brightness/contrast, and color transfer* from the target object of same semantic (category **people** in both the cases) in the second selfie. For color transfer, I computed statistics of the luminance and color temperature and used the histogram matching method.

*This way I had a composite in which the statistical features of one of the subjects was completely different compared to its background (this composite acts as a substitute for an extracted subject from the first selfie, and pasted into another, at the best possible location).*

*Images from the Dataset:*

![1](https://user-images.githubusercontent.com/41862477/49573700-aa0fad00-f964-11e8-8466-7bd4d7780fa5.JPG)
![2](https://user-images.githubusercontent.com/41862477/49573701-aa0fad00-f964-11e8-84a9-2ca1f3d8da40.JPG)
![3](https://user-images.githubusercontent.com/41862477/49573703-aaa84380-f964-11e8-8225-2d0c02ef89ac.JPG)

COCO dataset has originally around 40,000 images in which category **people** covered a significant portion of the image. I tried my best to ensure that, the edited images are neither arbitrary, nor unrealistic in color and tone by carrying out the editing of the  appearance of the subject very carefully, but unfortunately, there were still a few images that one would ideally not consider to keep in his/her dataset while training the Deep Learning model (**Garbage In Garbage Out** is one of the most common phrases in Machine Learning community). It will also be difficult for one to go through every image and filter out the bad composite since it is a repulsive and time-consuming task.

> *So, to overcome that issue, I trained a **discriminative** Deep CNN which learned the perception of visual realism in terms of **color, lighting and texture compatibility**, from a large amount of **unlabelled data**. It was able to make a proper distinction between the natural images and automatically generated image composites.* By using this model, I was able to discard very unnatural composites. Finally, we have a dataset in which the edited object still looks plausible but does not match the background context.

### Training ConditionalGAN:

Given the dataset and final aim in mind, I reduced the whole task to **Image to Image translation** problem (Motivated by this astounding paper [Pix2pix](https://arxiv.org/pdf/1611.07004.pdf)). In analogy to automatic language translation, automatic image-to-image translation refers to the task of translating one possible representation of an image into another, given sufficient training data. I deployed **Conditional Generative Adversarial networks** not only to learn the mapping from the input image to the output image but also to learn a loss function to train this mapping. This generic approach alleviated the problems that would traditionally require very different loss formulations (Image compositing has already been thoroughly investigated by Image Processing community, and there are numerous proposals to get some decent results by following some specific, well-defined approach. However, they aren't scalable at all for obvious reasons).

## Pix2pix

> ***If you are unfamiliar with GANs, I would request you to look into my tutorials (GAN_Zoo) and then proceed forward.***

From now on, I will only be giving you the overview of **Pix2pix** paper and the changes (mostly in the architecture and loss function) that I made to make the model work properly, remaining details are nearly the same. Like other GANs, Conditional GANs also have one discriminator (or critic depending on the loss function you are using) and one generator, and it tries to learn a conditional generative model which makes it suitable for image-to-image translation tasks, where we condition on an input image and generate a corresponding output image. 

> If mathematically expressed, CGANs learn a mapping from observed image X and random noise vector z, to y, G : {x, z} → y. The generator G is trained to produce outputs that cannot be distinguished from **real** images by an adversarially trained discriminator, D, which is again optimized to do as well as possible at identifying the generator’s **fakes**.

### Loss Function

The objective of a conditional GAN can be expressed as:

> **```LcGAN (G, D) = Ex,y (log D(x, y)) + Ex,z (log(1 − D(x, G(x, z)))```** where G tries to minimize this objective against an adversarial D that tries to maximize it, i.e. 

> **```G = arg min(G)max(D) LcGAN (G, D)```**. It is beneficial to mix the GAN objective with a more traditional loss, such as L1 distance. **```L(G) = Ex,y,z ( ||y − G(x, z)|| )```**.

Without z, the net could still learn a mapping from x to y, but would produce deterministic outputs, and therefore fail to match any distribution other than a **delta function**. Instead, the authors of Pix2pix provided noise only in the form of **dropout**, applied on several layers of the generator at **both training and test time**.

The objective (Min-Max) that I mentioned above was used in the original paper when GAN was first proposed by **Ian Goodfellow** in 2014, but unfortunately, it didn't perform well due to vanishing gradients problems (I won't go into much more depth here; refer to my GAN tutorials for more details). Since then, there has been a lot of development, and many researchers have proposed different kinds of loss functions (LS-GAN, WGAN, WGAN-GP) to overcome these issues. Authors of this paper used **Least-square** objective function while running their optimization process. 

None of the loss functions are optimal in every scenario, it's always task dependent (maybe WGAN performs better than LS-GAN in one task, while the other way around, in some other). Moreover, there was a recent paper by **Google** which also addressed this issue and showed that all loss functions give you nearly the same results, with the only condition that you need to do extensive hyper-parameter optimization. Training GANs is very tricky, and it will never work in the first attempt. You might even have to look into the capacity of your architecture. There was this, one recent theoretical paper on GANs by [Sanjeev Arora](https://arxiv.org/abs/1706.08224) in which he mentioned that the generator's capacity should be as large as twice the capacity of the discriminator. 

> From the above discussion, we can conclude that one spends a lot of time tweaking the hyper-parameters to make the GANs work properly.

### Network Architecture

#### Generator: 

In Image-to-image translation problems, we map a high resolution input grid to a high resolution output grid. Both are renderings of the same underlying structure with the only difference in the surface appearance. The authors designed the generator architecture around these considerations. They used an encoder-decoder network in which the input is passed through a series of layers that progressively downsample, until a bottleneck layer, at which point the process is reversed. 

> To preserve the low-level details, skip connections are used.  Specifically, skip connections are added between each layer i and layer n − i, where n is the total number of layers. Each skip connection simply concatenates all channels at layer i with those at layer n − i.

```Architecture: 
Encoder:  C(64, 1) - C(64, 1) - C(64, 2) - C(128, 2) - C(256, 2) - C(512, 2) - C(512, 2) - C(512, 2) - C(512, 2)
Decoder:  CD(512, 2) - CD(512, 2) - CD(512, 2) - CD(256, 2) - CD(128, 2) - CD(64, 2) - CD(64, 2) - CD(64, 1) - CD(3, 1)
```

#### Discriminator:

The GAN discriminator relies on high-frequency structure term to force low-frequency correctness. In order to model high-frequencies, it is sufficient to restrict the attention to the structure in local image patches. Therefore, discriminator architecture was termed PatchGAN – that only penalizes structure at the scale of patches. This discriminator tries to classify if each N × N patch in an image
is real or fake. Run this discriminator convolutionally across the image, and average all responses to provide the ultimate output of D.

> Instead of using vanilla convolutional layers to increase the receptive field (or decrease the size of the image), use Res-Block or WideRes-Block with the instance normalization sliced between the layers. This way, we can get an increment in the capacity of the discriminator (which was necessary in this case) which ultimately resulted in better composites. 

The original patch-gan was giving some weird artifacts due to the fact that, it was optimized for some different task (Authors pointed out that receptive field of 70 * 70 was giving best results, compared to smaller or larger recpetive field). It is always preferred to use Instance normalization compared to Batch Normalization in the case of GANs, but in my case both were giving nearly same results and bacth normalization was faster (batch size greater than 1, so GPU computations can be done much more efficiently).

> Patch GANs discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter.

```Architecture: WRB64 - WRB128 - WRB256 - WRB256```

***

## Results

![1](https://user-images.githubusercontent.com/41862477/49601764-bfa8c500-f9ac-11e8-83ec-355d77cb4d5c.JPG)
![2](https://user-images.githubusercontent.com/41862477/49601766-bfa8c500-f9ac-11e8-92a2-19625692fcd8.JPG)
![3](https://user-images.githubusercontent.com/41862477/49601767-c0415b80-f9ac-11e8-8546-392744c8a696.JPG)

![out_1](https://user-images.githubusercontent.com/41862477/49601769-c0415b80-f9ac-11e8-8931-14423a5c9016.JPG)
![out_2](https://user-images.githubusercontent.com/41862477/49601770-c0415b80-f9ac-11e8-8721-55d6ed024eb1.JPG)
![out_3](https://user-images.githubusercontent.com/41862477/49601771-c0d9f200-f9ac-11e8-9f16-40cb608648b4.JPG)

***

### ***Thanks for going through this post!***
