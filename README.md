# Image Compositing

#### *Note: This work was done by me during the internship at Samsung Research Institute last summer. It addresses an intriguing problem which I think is quite relevant from an industrial research point of view, especially for companies that are working on AI-powered smartphones.*

## Aim
**In short, the aim was to generate aesthetically pleasing, photorealistic composite by merging two (or more) selfies.**

*You might be thinking, what does that even mean?*

Let me try to make things simple and explain it to you in layman language with the help of an example. Suppose we have two selfies, so what we want to do is, extract any one individual (usually pointed out by the user by a click) from the first selfie and paste it automatically at the best possible location in the second selfie, so that the composite doesn't look weird. *I will be using the word subject to refer to the individual extracted from the first selfie, from now on to keep the discussion manageable.* Now, if you start thinking about it, this seems to be a very complicated task since finding the best among many of the possible salient regions in the second selfie, while keeping in mind the spatial relationship between the subject and background (second selfie) is not a trivial thing to do. The algorithm should be very robust to scale and rotational invariance. 

That was the first part of the problem, coming to the second - The lighting condition of the two selfies can be completely different (one can be in broad light, another one in backlight or one can be in bright daylight and another one in low dim night light). So, we need to match the lighting conditions of the background and foreground in a way that the composite looks natural. It should be clear by looking at the composite that both (or more than two) individuals were present while clicking the selfie.

***So, this was the final aim and hope that I was able to explain it clearly.*** 

## Motivation

One can think of many potential use cases of it, but I would like to state few, which I can think off the top of head:

**1) Movie Industry** can use it to create their CGI scenes.  **2) Personal use:** People can create and share their memories just like collage. **3) Reduced Computations & Low Inference Time** These all can be achieved by Photoshop but that would require a whole lot of manual labor and huge computations.  

## Results
![1](https://user-images.githubusercontent.com/41862477/49270708-952ca880-f490-11e8-86a7-e9b5e2e483ad.JPG)
![out_1](https://user-images.githubusercontent.com/41862477/49270712-95c53f00-f490-11e8-97c6-878247047365.JPG)

![2](https://user-images.githubusercontent.com/41862477/49270709-95c53f00-f490-11e8-8ca8-384542f324dc.JPG)
![out_2](https://user-images.githubusercontent.com/41862477/49270713-965dd580-f490-11e8-92dd-cdd37dd2e3be.JPG)

![3](https://user-images.githubusercontent.com/41862477/49270711-95c53f00-f490-11e8-99cc-47ec16ddc6d6.JPG)
![out_3](https://user-images.githubusercontent.com/41862477/49270714-965dd580-f490-11e8-8fe9-f889bd42fd2f.JPG)
