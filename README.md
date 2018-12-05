# Image Compositing

#### *Note: This work was done by me during the internship at Samsung Research Institute last summer. It addresses an intriguing problem which I think is quite relevant from an industrial research point of view, especially for companies that are working on AI-powered smartphones.*

## Aim
*In short, the aim was to generate aesthetically pleasing, photorealistic composite by merging two (or more) selfies.*

You might be thinking, what does that even mean?

Let me try to make things simple and explain it to you in layman language with the help of an example. Suppose we have two selfies, so what we want to do is, extract any one individual (usually pointed out by the user by a click) from the first selfie and paste it at the best possible location in the second selfie, so that the composite doesn't look odd when the subject gets placed. I will be using the word subject to refer to the individual extracted from the first selfie from now on to keep the discussion manageable. Now, this is a very complicated task since finding the best among many of the possible salient regions in the second selfie, while keeping in mind the spatial relationship between the subject and background (second selfie) is not a trivial thing to do. The algorithm should be very robust to scale and rotational invariance of the subject. After we are done with this part, we have another obstacle in our way. The lighting condition of the two selfies can be completely different (one can be clicked in hot temperature, other one in cold or one can be in bright day light and other one in low dim night light). So, we need to somehow match the lighting conditions of the background and foreground in a way that it looks like that both (or more than two) subjects were present while clicking the selfie. Hope, I was able to explain you the aim of the project. 


## Results
![1](https://user-images.githubusercontent.com/41862477/49270708-952ca880-f490-11e8-86a7-e9b5e2e483ad.JPG)
![out_1](https://user-images.githubusercontent.com/41862477/49270712-95c53f00-f490-11e8-97c6-878247047365.JPG)

![2](https://user-images.githubusercontent.com/41862477/49270709-95c53f00-f490-11e8-8ca8-384542f324dc.JPG)
![out_2](https://user-images.githubusercontent.com/41862477/49270713-965dd580-f490-11e8-92dd-cdd37dd2e3be.JPG)

![3](https://user-images.githubusercontent.com/41862477/49270711-95c53f00-f490-11e8-99cc-47ec16ddc6d6.JPG)
![out_3](https://user-images.githubusercontent.com/41862477/49270714-965dd580-f490-11e8-8fe9-f889bd42fd2f.JPG)
