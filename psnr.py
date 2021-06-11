import torch
import torch.nn as nn

def PSNR(generate_image, target_image):
    #mse = nn.MSELoss(generate_image, target_image, reduction='none')
    mse = torch.mean(torch.square(generate_image - target_image))
    max_pixel = 1.0
    psnr = 10 * torch.log10((max_pixel ** 2) / mse)
    return psnr