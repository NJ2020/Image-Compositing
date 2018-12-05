# Image Compositing

## Abstract
*The aim of the project is to generate aesthetically pleasing, photorealistic composite by merging two (or more) selfies.*
As you noted, it's difficult to describe the aim in one line. So, let's break things up in small pieces, and spend some time together to figure out every minute detail about it. Suppose we have any two selfies. What we want to do is extract one subject (can be anyone pointed out by the user) from the first selfie and paste it at the best possible location in the second selfie (the composite shouldn't look odd when the subject from 1st selfie gets placed in the second one). Now, this is a challenging task since finding one of the many possible salient regions for the subject in the second selfie, while keeping in mind the spatial relationship is not a trivial thing to do. The algorithm should also be robust to scale and rotational invariance of the extracted subject from the first selfie. 


## Results
![1](https://user-images.githubusercontent.com/41862477/49270708-952ca880-f490-11e8-86a7-e9b5e2e483ad.JPG)
![out_1](https://user-images.githubusercontent.com/41862477/49270712-95c53f00-f490-11e8-97c6-878247047365.JPG)

![2](https://user-images.githubusercontent.com/41862477/49270709-95c53f00-f490-11e8-8ca8-384542f324dc.JPG)
![out_2](https://user-images.githubusercontent.com/41862477/49270713-965dd580-f490-11e8-92dd-cdd37dd2e3be.JPG)

![3](https://user-images.githubusercontent.com/41862477/49270711-95c53f00-f490-11e8-99cc-47ec16ddc6d6.JPG)
![out_3](https://user-images.githubusercontent.com/41862477/49270714-965dd580-f490-11e8-8fe9-f889bd42fd2f.JPG)
