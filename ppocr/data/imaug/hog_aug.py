# https://blog.csdn.net/weixin_44791964/article/details/103549605
# https://www.kaggle.com/code/brendan45774/hog-features-histogram-of-oriented-gradients#Hog-Images

from skimage.feature import hog
from skimage.exposure import rescale_intensity
import cv2
import numpy as np
from PIL import Image

__all__  = ['HOGaug']

class HOGaug(object):
    def __init__(self, **kwargs):
        pass

    def save(self, img, name):
        new_p = Image.fromarray(img)
        print(new_p.mode)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(name)

    def __call__(self, data):
        img = data['image']
        # print('call HOG', type(img), img.shape)
        # self.save(img, 'hog_1.jpg')
        resized_image = img
        resized_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, _hog = hog(resized_image, orientations=9, pixels_per_cell=[8,8],
                   cells_per_block=[2,2], visualize=True) #, channel_axis=2)
        _hog = rescale_intensity(_hog, out_range=(0, 255))
        # print(_hog.shape)
        h, w = _hog.shape
        color_img = np.zeros([h, w, 3])
        color_img[:, :, 2] = _hog  # In opencv images are BGR
        # color_img = color_img * 255
        color_img = color_img.astype(np.uint8)
        # print('call HOG(2)', type(color_img), color_img.shape)
        # self.save(color_img, 'hog_2.jpg')
        data['image'] = color_img
        return data
