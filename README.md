# Image Compositing

#### *Note: This work was done by me during the internship at Samsung Research Institute, last summer. It addresses an intriguing problem which I think is quite relevant from an industrial research point of view, especially for companies that are working on AI-powered smartphones.*

## Aim
>**In short, the aim was to generate aesthetically pleasing, photorealistic composite by merging two (or more) selfies.**

*You might be thinking, what does that even mean?*

Let me try to make things simpler and explain it to you in layman language with the help of an example. Suppose we have two selfies, so what we want to do is, extract any one individual (usually pointed out by the user through a click) from the first selfie and paste it automatically at the best possible location in the second selfie, without making the composite look unnatural. 

>*I will be using the word subject to refer to the individual extracted from the first selfie from now on, to keep the discussion manageable.* 

Now, if you start thinking about it, this seems to be a very complicated task since finding the best among many of the possible salient regions in the second selfie, while keeping in mind the spatial relationship between the subject and background (second selfie) is not a trivial thing to do. The algorithm should be very robust to scale and rotational invariance. 

Assuming we somehow managed to do the first part, still, we have another major challenge to solve - The lighting condition of the two selfies can be completely different (one can be in broad light, another one in backlight or one can be in bright daylight and another one in low dim nightlight). So, we also need to match the lighting conditions of the background (second selfie) and foreground (subject) in a way that the overall composite looks natural. It should be clear by looking at the composite that both the individuals were present while clicking the selfie.

>***So, this was the final aim and hope that I was able to explain it clearly.*** 

## Motivation

*One can think of many potential use cases of it, but I would like to state few which I can think off, the top of my head. We all can do these things mentioned below in Photoshop but that would require a lot of manual labor, and considerable inference time.*

> - People can create and share their memories just like a collage for their **personal use.**

> - **Anime industry** can use it to create CGI scenes by skipping the intermediate step of the 3D rendering of an object.   

## Problem Breakdown
The exciting part about the project was that neither has this Image to Image translation problem been explored much by Deep Learning
nor are there any publicly available datasets for Image Compositing. 

As I started pondering on how to approach this complex task at hand, I decided to divide the whole project into two parts as already described above in much detail. 

> **1)** Running an **instance segmentation** over the first selfie to extract the relevant subject from it and then, to automatically detect the most consistent region among many in the second image, where subject from the first image would not look out of place.

> **2)** Using **Conditional GANs** to match the lighting conditions and other statistical features of the foreground and background to give it a photo-realistic touch.

Due to time constraints, I took a slight detour, spent a considerable amount of time creating my custom dataset using COCO segmented data, to be used as an input to the latter half. Creating a meaningful dataset while keeping the final aim of the project in mind
was the most challenging task since it was the only way to get the best possible proxy for the first part. Following is a brief overview of what I did: 

> **1)** Picked any two random images from the whole dataset (original COCO dataset) in which the category people covered a significant portion of the image. 

> **2)** Masked a random subject from the first image and applied Image processing techniques like histogram equalization, random perturbation in brightness/contrast, transferring the color of the second image to the masked subject of the first image and, few other techniques so to say. 

*This way I had a composite in which the statistical features of one of the subjects was completely different compared to its
background (this acts as a proxy for a extracted subject picked from the first selfie pasted into another).*





## Results
![1](https://user-images.githubusercontent.com/41862477/49270708-952ca880-f490-11e8-86a7-e9b5e2e483ad.JPG)
![out_1](https://user-images.githubusercontent.com/41862477/49270712-95c53f00-f490-11e8-97c6-878247047365.JPG)

![2](https://user-images.githubusercontent.com/41862477/49270709-95c53f00-f490-11e8-8ca8-384542f324dc.JPG)
![out_2](https://user-images.githubusercontent.com/41862477/49270713-965dd580-f490-11e8-92dd-cdd37dd2e3be.JPG)

![3](https://user-images.githubusercontent.com/41862477/49270711-95c53f00-f490-11e8-99cc-47ec16ddc6d6.JPG)
![out_3](https://user-images.githubusercontent.com/41862477/49270714-965dd580-f490-11e8-8fe9-f889bd42fd2f.JPG)
