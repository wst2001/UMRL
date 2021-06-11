import torch
from torchvision import models, transforms
from myutils import vgg16
from PIL import Image
import os
vgg = vgg16.Vgg16()

img = Image.open("./facades/0.jpg").convert('RGB')
width, height = img.size
img = img.crop((0, 0, width // 2, height))
#img = img.crop((width // 2, 0, width, height))
img = transforms.ToTensor()(img)
img = img.unsqueeze(dim=0)
features = vgg(img)

to_img = transforms.ToPILImage()
cnt = 0
for fea in features:
    tmp = fea[:, 0, :, :] - fea[:, 1, :, :]
    fea_img = to_img(tmp.squeeze())
    fea_img.save("./check/%s.png" % (cnt))
    cnt += 1

