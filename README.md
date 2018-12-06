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

- Running an **instance segmentation** over the first selfie to extract the relevant subject from it and then, to automatically place it at the most suitable position in the second selfie, where the subject would not look out of place.
- Using **Conditional GANs** to match the lighting conditions and other statistical features of the foreground (subject) and background (second selfie) to give it a photo-realistic touch.

> Due to time constraints, I took a slight detour, spent a considerable amount of time creating my custom dataset using **COCO** data, which I can use as an input to the latter half. Creating a meaningful dataset while keeping the final aim of the project in mind was the most challenging task since it was the only way to get the best possible proxy for the first part. Following is a brief overview of what I did: 

### Dataset Creation

- Picked any two random images from the whole dataset (original COCO) in which the category people covered a significant portion of the image. 
- Masked a subject from the first image and edited its appearance using Image processing techniques like *random perturbation in brightness/contrast, and color transfer* from the target object of same semantic (category **people** in both the cases) in the second selfie. For color transfer, I computed statistics of the luminance and color temperature, and used the histogram matching method.

*This way I had a composite in which the statistical features of one of the subjects was completely different compared to its background (this composite acts as a substitute for an extracted subject from first selfie, pasted into another, at the best possible location).*

*Images from the Dataset:*

![1](https://user-images.githubusercontent.com/41862477/49573700-aa0fad00-f964-11e8-8466-7bd4d7780fa5.JPG)
![2](https://user-images.githubusercontent.com/41862477/49573701-aa0fad00-f964-11e8-84a9-2ca1f3d8da40.JPG)
![3](https://user-images.githubusercontent.com/41862477/49573703-aaa84380-f964-11e8-8225-2d0c02ef89ac.JPG)

COCO dataset has originally around 40,000 images in which category **people** covered a significant portion of the image. I tried my best to ensure that, the edited images are neither arbitrary nor unrealistic in color and tone by carrying out the editing of the  appearance of the subject very carefully, but unfortunately there were still a few images that one would ideally not consider to keep in his/her dataset while training the final Deep Learning model (**Garbage In Garbage Out** is one of the most common phrases in Machine Learning community). It will also be difficult for one to go through every image and filter out the bad composite since it is a repulsive and time-consuming task.

> *For that, I trained a **discriminative** Deep CNN which learned the perception of visual realism in terms of **color, lighting and texture compatibility**, from a large amount of **unlabelled data**. It was able to make a proper distinction between the natural images and automatically generated image composites.* By using this model, I was able to discard very unnatural composites. Finally, we have a dataset in which the edited object still looks plausible but does not match the background context.

### Training ConditionalGAN:

Given the dataset and final aim in mind, I reduced the whole task to **Image to Image translation** problem (Motivated by this astounding paper [Pix2pix](https://arxiv.org/pdf/1611.07004.pdf)). In analogy to automatic language translation, automatic image-to-image translation refers to the task of translating one possible representation of an image into another, given sufficient training data. I deployed **Conditional Generative Adversarial networks** not only to learn the mapping from the input image to the output image but also to learn a loss function to train this mapping. This generic approach alleviated the problems that would traditionally require very different loss formulations (Image compositing has already been thoroughly investigated by Image Processing community, and there are numerous proposals to get some decent results by following some specific, well-defined approach. However, they aren't scalable at all for obvious reasons).





## Results
![1](https://user-images.githubusercontent.com/41862477/49270708-952ca880-f490-11e8-86a7-e9b5e2e483ad.JPG)
![out_1](https://user-images.githubusercontent.com/41862477/49270712-95c53f00-f490-11e8-97c6-878247047365.JPG)

![2](https://user-images.githubusercontent.com/41862477/49270709-95c53f00-f490-11e8-8ca8-384542f324dc.JPG)
![out_2](https://user-images.githubusercontent.com/41862477/49270713-965dd580-f490-11e8-92dd-cdd37dd2e3be.JPG)

![3](https://user-images.githubusercontent.com/41862477/49270711-95c53f00-f490-11e8-99cc-47ec16ddc6d6.JPG)
![out_3](https://user-images.githubusercontent.com/41862477/49270714-965dd580-f490-11e8-8fe9-f889bd42fd2f.JPG)
