import numpy as np
import torch
from skimage.metrics import structural_similarity

def SSIM(generate_image, target_image):
    generate_image = generate_image.cpu().squeeze().detach().permute(1, 2, 0).numpy()
    target_image = target_image.cpu().squeeze().detach().permute(1, 2, 0).numpy()
    ssim = structural_similarity(generate_image, target_image, data_range=target_image.max()-target_image.min(), multichannel=True)
    return ssim
'''def SSIM(generate_image, target_image):
    generate_image = generate_image.cpu().squeeze().detach().permute(1, 2, 0).numpy()
    target_image = target_image.cpu().squeeze().detach().permute(1, 2, 0).numpy()
    generate_image = tf.convert_to_tensor(generate_image)
    target_image = tf.convert_to_tensor(target_image)
    ssim = tf.image.ssim(generate_image, target_image, max_val=1.0)
    return ssim
'''