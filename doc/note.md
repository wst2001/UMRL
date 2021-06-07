# UMRL

## 1 Introduction

single image de-rain

prior information

1. high frequency details: remove important parts

2. artifacts

rain streak location information

Unet + skip connection

uncertainty map

discard the rain content learned by layer “l” if the confidence value is low in the uncertainty map.



cycle spinning

reduce artifacts

## 2 Background and Related Work

GMM

low-rank representation

CNN

GAN

RNN

## 3 Proposed Method

### 3.1 UMRL

confidence score: how much the network is certain about the residual value

![image-20210605122843687](C:\Users\wst\AppData\Roaming\Typora\typora-user-images\image-20210605122843687.png)

![image-20210605123003939](C:\Users\wst\AppData\Roaming\Typora\typora-user-images\image-20210605123003939.png)

use the information about the location in image where network might go wrong in estimating the residual value

confidence map at different scales

This information is then fed back to the subsequent layers so that the network can learn the residual value at each location

loss:

![image-20210605123518554](C:\Users\wst\AppData\Roaming\Typora\typora-user-images\image-20210605123518554.png)

### 3.2 Cycle Spinning

shift the image cyclically

de-rain the shifted images

inverse shift and average them

