# https://blog.csdn.net/weixin_44791964/article/details/103549605
# https://www.kaggle.com/code/brendan45774/hog-features-histogram-of-oriented-gradients#Hog-Images

from skimage.feature import hog
from skimage.exposure import rescale_intensity
import cv2
import numpy as np
from PIL import Image

__all__  = ['AddOneDimaug']

class AddOneDimaug(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        image = data['image']
        image = image.transpose((1, 2, 0))
        h, w, _ = image.shape
        empyt_img = np.zeros((h, w, 1), dtype=np.float32)
        enhanced_image = np.dstack((image, empyt_img)).transpose((2, 0, 1))
        data['image'] = enhanced_image
        return data
 

          

