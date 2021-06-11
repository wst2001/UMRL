import tensorflow as tf
import numpy as np
import torch
def SSIM(generate_image, target_image):
    generate_image = generate_image.detach().permute(1, 2, 0).numpy()
    target_image = target_image.detach().permute(1, 2, 0).numpy()
    generate_image = tf.convert_to_tensor(generate_image)
    target_image = tf.convert_to_tensor(target_image)
    ssim = tf.image.ssim(generate_image, target_image, max_val=1.0)
    return ssim
